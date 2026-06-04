# Pickle → DuckDB migration design

Status: **draft, for review**. No code touched yet.
Baseline commit: `0392812`. WIP attempt is preserved in `stash@{0}` ("wip duckdb attempt before restart").

---

## 1. Why we're doing this

`process_building_parts` and its downstream consumers hold whole `GeoDataFrame`s in RAM. The pickles are the *symptom* — the actual cost is:

1. **Producer side** (`building_inference.py:739–763`): the existing code already serialises sbr then br to keep peak at ~3 frames instead of 4, but at the worst point a full `sbr_results` and `br_parts` list both live in RAM. On Barcelona-sized inputs this is multi-GB.
2. **Aggregation tail** (`building_inference.py:688–728`): every zone's pickle is read back, downcast, and `pd.concat`ed in batches of 200. The whole municipality must fit in memory at the end.
3. **Consumer side** (`mergers.py:797–897`): receives `(sbr_results, br_results)` as in-memory frames and does multiple `merge` + `groupby` + `apply` passes. This is the dominant RAM peak in practice — it sits on top of everything else `mergers.py` is already holding (`building_gdf_`, `buildings_CAT`, ...).

Returning DataFrames between functions is the part we need to kill. DuckDB lets us:

- write **per-zone results directly into a single DB file**, no per-zone pickle, no final concat.
- let `mergers.py` do `merge`/`groupby` as **SQL against the DB**, never materialising the whole frame.
- materialise small pandas slices only at the last possible moment (e.g. when producing the final municipality geojson/parquet).

---

## 2. Inventory of pkl sites (baseline = commit `0392812`)

### 2.1 Package internals (`hypercadaster_ES/`)

**Writes** — all in `building_inference.py`:

| Line | File pattern | What it stores |
|---|---|---|
| 490 | `cache/{code}_aux1.pkl` | `gdf_global` — building parts + nearby_buildings/parcels flags |
| 504 | `cache/{code}_aux2.pkl` | `gdf_footprints_global` — per-building floor footprints |
| 2032–2035 | `cache/{zone_reference}_{sbr,br}_results.pkl` | per-zone inference output (atomic write via .tmp) |
| 743–763 | `{code}_{sbr,br}_results.pkl` | municipality-level concatenated output (zstd) |

**Reads of the above** — inside the package:

| Site | Purpose |
|---|---|
| `building_inference.py:82–86` | `_read_compressed_pickle` — detects gzip vs zstd magic bytes |
| `building_inference.py:440–441` | municipality-cache reuse path |
| `building_inference.py:492, 506` | per-municipality aux cache reuse |
| `building_inference.py:715–716` | per-zone cache read in the aggregation loop |
| `building_inference.py:2044–2045` | per-zone cache reuse inside `process_zone` |

**Public readers** — `utils.py:4220–4230`:

| Function | File pattern | Producer |
|---|---|---|
| `read_br_inferred_indicators` | `{code}_br_results.pkl` (gzip) | this repo (building_inference.py) |
| `read_sbr_inferred_indicators` | `{code}_sbr_results.pkl` (gzip) | this repo (building_inference.py) |
| `read_addresses_indicators` | `{code}_no_inference.pkl` (gzip) | **external pipeline only** — no writer in this repo |

**Consumer**: `mergers.py:797` unpacks `(sbr_results, br_results)` from `process_building_parts` and runs merges/groupbys against the returned pandas frames. `hc.merge()` itself returns one giant in-memory GeoDataFrame to its caller.

### 2.2 Downstream pickle sites in `examples/`

Every consumer of `hc.merge()` re-pickles its output. The "return whole pandas frame" pattern is therefore not contained to the package — it propagates all the way to user scripts:

| File | Line | Pickle |
|---|---|---|
| `buildings_complete_inference.py` | 50, 70, 78, 98, 100 | `{code}_complete_inference.pkl`, combined `{codes}_complete_inference.pkl` |
| `buildings_without_inference.py` | 72 | output gdf |
| `buildings_without_inference_from_boundary_box.py` | 115 | output gdf |
| `buildings_batched_processing.py` | 145 | per-batch output gdf |
| `multi_municipality_basic.py` | 56 | combined gdf |
| `generate_buildings_and_census_geojsons.py` | 54 | reads `{codes}_only_addresses.pkl` |
| `dataset_for_irec_clustering.py` | 41, 154 | reads `{codes}.pkl`, writes `gdf_irec` |
| `dataset_for_understanding_facade_geom.py` | 38 | reads `{codes}.pkl` |
| `cups_lme.py` | 227 | reads `08900_complete_inference.pkl` |
| `test_single_zone_shadow.py` | 43, 44 | reads `aux1`/`aux2` |

Implication: `hc.merge()`'s return contract is the load-bearing one. Until that stops being "a single GeoDataFrame", every example has to materialise the whole municipality just to save it again. See §4 — the `ResultsHandle` model needs to apply to `hc.merge()` too, not just `process_building_parts`.

---

## 3. Proposed layout

### 3.1 One DuckDB file per `results_dir`

```
{results_dir}/
    building_results.duckdb        # single-writer, multi-reader inside a run
    cache/                         # only for the two municipality aux gdfs (see §3.5)
```

Rationale: one DB per municipality is the cleanest aggregation unit. Multiple municipalities each get their own DB; we don't try to multiplex.

### 3.2 Tables

```sql
-- per-zone inference outputs, written incrementally as zones complete
CREATE TABLE sbr_results (
    cadaster_code     VARCHAR NOT NULL,
    zone_reference    VARCHAR NOT NULL,
    building_reference VARCHAR NOT NULL,
    single_building_reference VARCHAR,
    -- ~N sbr__* columns, dtype-mapped from current pandas frames
    ...
);
CREATE INDEX sbr_results_br ON sbr_results(building_reference);

CREATE TABLE br_results (
    cadaster_code     VARCHAR NOT NULL,
    zone_reference    VARCHAR NOT NULL,
    building_reference VARCHAR PRIMARY KEY,
    -- ~N br columns
    -- list-typed columns (building_spaces_*) stored as DuckDB LIST<...>
    ...
);

-- progress / idempotency ledger — replaces the per-zone pickle existence check
CREATE TABLE zone_status (
    zone_reference VARCHAR PRIMARY KEY,
    state          VARCHAR NOT NULL,       -- 'ok' | 'empty' | 'err'
    finished_at    TIMESTAMP DEFAULT now(),
    error          VARCHAR
);
```

`{code}_{sbr,br}_results.pkl` and the per-zone pickles all become rows in `sbr_results` / `br_results` keyed by `cadaster_code` (and partitioned by `zone_reference` inside the table). The "did this zone already complete?" check becomes a `SELECT state FROM zone_status WHERE zone_reference = ?`.

### 3.3 Geometry — DuckDB spatial extension (decided)

Native `GEOMETRY` columns via `INSTALL spatial; LOAD spatial;`. Means:

- `ST_GeomFromWKB` / `ST_AsWKB` on insert/read; shapely round-trip only when a caller actually needs shapely objects.
- SQL-side spatial ops (`ST_Intersects`, `ST_Buffer`, `ST_Area`) become available — useful for the `mergers.py` step 5 work where we currently do shapely-per-row.
- CRS is **not stored by the extension** — record it once per table in a small `_meta` table (`table_name, crs_epsg`) so readers can re-attach CRS when handing geometries back to geopandas.
- Hard dep on `duckdb >= 0.10` with the spatial extension. Add `INSTALL spatial; LOAD spatial;` to `open_db()` so the extension is auto-loaded on every connection.

### 3.4 List / dict columns

`br_results` contains list-valued columns (`building_spaces_*`) and `mergers.py` constructs dict-valued columns (one dict per `building_reference`). Options:

- **LIST<STRUCT<...>>** — typed, queryable, native. Best for the `building_spaces_*` lists where the inner schema is known.
- **JSON (VARCHAR)** — opaque blob. Cheapest to migrate, works for the heterogeneous dicts in `mergers.py`.

**Recommendation**: LIST<STRUCT> where the schema is uniform; JSON for the genuinely heterogeneous sbr_multi dicts.

### 3.5 Two aux gdfs

`{code}_aux1.pkl` (`gdf_global`) and `{code}_aux2.pkl` (`gdf_footprints_global`) are **per-municipality scratch** that is reused on resume but never read by anything outside `process_building_parts`. Two options:

- **(a)** Keep as pickle. They're written/read once each and are scratch — converting them costs effort with no RAM win, since they're loaded whole either way.
- **(b)** Move to the same `building_results.duckdb` under `aux1` / `aux2` tables, and switch consumers to scan-on-demand (e.g. iterate by zone instead of holding the full frame).

**Decision: (b), in this pass.** Critical worker-model consequence below.

Today, workers `fork()` from the parent and inherit `gdf_footprints_global` + `_ZONE_WORKER_STATE` via copy-on-write (see the comment block at `building_inference.py:563–588`, which explicitly warns this is the reason RAM stays bounded). If we delete the in-memory `gdf_footprints_global` and have workers query the DB instead, we lose copy-on-write inheritance but **also** lose the reason the parent has to hold the whole frame.

New worker contract:

- **Parent**: opens DB, writes `aux_building_parts` / `aux_footprints` once. Does **not** keep them in memory afterwards.
- **Each worker**: opens its **own** read-only DuckDB connection (`duckdb.connect(db_path, read_only=True)`). For each task, runs `SELECT * FROM aux_footprints WHERE building_reference IN (refs_for_this_zone)` (or a spatial pre-filter for shadow-casting neighbours). Only ever holds the slice it currently needs.
- `_ZONE_WORKER_STATE` shrinks to: `{ "db_path", "results_dir", "plots", "zone_cat_lookup", ... }`. CAT lookup is small (per-zone <1 KB slices) so it stays inherited.

This is step 4 in the new ordering (§6).

---

## 4. New API surface

Two layers change:

- **Internal** (`process_building_parts`): returns a `ResultsHandle` instead of `(sbr_results, br_results)` tuples.
- **Public** (`hc.merge()`): returns a `MunicipalityHandle` instead of a giant `GeoDataFrame`. This is the load-bearing one — every `examples/*.py` does `gdf = hc.merge(...); gdf.to_pickle(...)`. Until `hc.merge()` stops returning whole frames, the user-facing RAM peak doesn't move regardless of what we do inside the package.

```python
# building_inference.py
def process_building_parts(..., results_dir, ...) -> ResultsHandle:
    """Returns a lightweight handle, not pandas frames."""

class ResultsHandle:
    """Cheap wrapper around a duckdb db_path + cadaster_code.

    Opens a new short-lived connection per query rather than holding one
    open — keeps the handle picklable across process boundaries if needed.
    """
    db_path: str
    cadaster_code: str

    def sbr_relation(self) -> duckdb.DuckDBPyRelation: ...
    def br_relation(self) -> duckdb.DuckDBPyRelation: ...
    def sbr_iter_batches(self, batch_size: int = 5000) -> Iterator[pd.DataFrame]: ...
    def br_iter_batches(self, batch_size: int = 5000) -> Iterator[pd.DataFrame]: ...
    # escape hatches — explicit so callers see the cost
    def sbr_to_pandas(self) -> pd.DataFrame: ...
    def br_to_pandas(self) -> pd.DataFrame: ...

# mergers.py / __init__.py
def merge(...) -> MunicipalityHandle:
    """Build the per-municipality output; return a handle, not a frame.

    The handle exposes the same `buildings` / `building_parts` /
    `addresses` views that callers used to receive as gdf columns.
    """

class MunicipalityHandle:
    db_path: str
    cadaster_code: str

    def buildings(self) -> duckdb.DuckDBPyRelation:    # one row per building
    def building_parts(self) -> duckdb.DuckDBPyRelation:  # one row per part
    def addresses(self) -> duckdb.DuckDBPyRelation:
    def iter_buildings(self, batch_size=5000): ...
    def to_geodataframe(self, kind="buildings") -> gpd.GeoDataFrame:  # escape hatch
    def to_geoparquet(self, path, kind="buildings"): ...
    def to_geojson(self, path, kind="buildings"): ...
```

**`mergers.py:797` becomes** — instead of unpacking gdf tuples and chaining `merge()` calls, it writes into the output tables:

```python
results = building_inference.process_building_parts(...)  # ResultsHandle
con = open_db(results_dir)  # spatial extension already loaded
con.register("building_gdf", building_gdf_)
con.execute("""
    INSERT INTO buildings
    SELECT b.gml_id, b.building_status, b.building_reference, ...,
           r.* EXCLUDE (building_reference)
    FROM building_gdf b
    LEFT JOIN br_results r USING (building_reference)
    WHERE r.cadaster_code = ?
""", [code])
```

The sbr aggregation at `mergers.py:810–897` (count-per-building, multi-vs-single split, dict construction, merge-back) becomes a single SQL pass: `GROUP BY building_reference` with conditional aggregation. Dict-valued columns become `MAP<VARCHAR, ...>` or JSON, written in one statement. **This is the single biggest RAM win** — today this code path holds the entire `sbr_results` in memory while building `sbr_multi` alongside.

**`read_*_indicators` public API**:

- `read_br_inferred_indicators`, `read_sbr_inferred_indicators` get a compatibility shim that queries the DB and returns pandas. Public signature unchanged — but downstream code still pays the materialisation cost.
- Add `br_handle()` / `sbr_handle()` siblings that return `ResultsHandle`. Recommend over the pandas readers in docstrings.
- `read_addresses_indicators` (currently reads `_no_inference.pkl`): becomes a projection of the same `buildings` table — see §7.1. No separate table.

**Examples** — since the user wants everything in DuckDB, the examples need updating too. Pattern:

```python
# old
gdf = hc.merge(...)
gdf.to_pickle(individual_filename, compression="gzip")
gdf = pd.read_pickle(individual_filename, compression="gzip")
hc.utils.plot_building_reference(gdf, '...')

# new
h = hc.merge(..., results_dir=wd)
# nothing to "save" — the DB at h.db_path is the artifact
hc.utils.plot_building_reference(h, '...')  # plot helpers take a handle
# escape hatch when something genuinely needs a gdf:
gdf = h.to_geodataframe()
```

Affected: all 10 example files listed in §2.2. They can be migrated in a single sweep once `MunicipalityHandle` lands, since they all follow the same pattern.

---

## 5. What goes away

- `_read_compressed_pickle` (building_inference.py:72–86)
- gzip-vs-zstd dual-decoder logic
- `_municipality_cache_present` size-check (building_inference.py:426–435) — replaced by `SELECT count(*) FROM zone_status WHERE state = 'ok'` or a sentinel row
- Atomic .tmp → rename dance (building_inference.py:2030–2035) — DuckDB transactions give us this for free
- Per-zone pickle read loop (building_inference.py:710–728) and the BATCH=200 concat — zones are already in the DB by the time the loop would run
- `_downcast_floats` (building_inference.py:680–686) — fixed-width column types in DuckDB give the same memory shape without the runtime cost

---

## 6. Migration order (smallest blast radius first)

Each step ends with a green run on a small municipality (recommend `25023` Alpicat — small, already in your example) before moving on, so we never have two breakages stacked.

1. **Foundations.** Add `duckdb` to `requirements.txt`, create `hypercadaster_ES/_db.py` with `open_db(results_dir)` (auto-loads spatial extension), `ensure_schema(con)`, `_meta` CRS table, helpers to convert geopandas ↔ DuckDB GEOMETRY. No call-site changes yet.
2. **Per-zone writer → DB.** Change `process_zone`'s atomic-pickle write (building_inference.py:2030–2035) to `INSERT` into per-zone shard DB `cache/{zone}.duckdb` (avoids single-writer contention — see §8). Parent attaches+unions on aggregation. *Keep* writing the pickle in parallel for one commit; assert DB and pickle agree on a small municipality.
3. **Per-zone reader → DB.** Replace the parent's read-back loop (building_inference.py:710–728) with `SELECT ... FROM attached zone DBs`, streamed. Drop the per-zone pickle write. Still write the two municipality-level pickles from the streamed query, so `read_*_indicators` keeps working.
4. **Aux frames → DB.** Move `gdf_global` / `gdf_footprints_global` writes (building_inference.py:490, 504) into `aux_building_parts` / `aux_footprints` tables. Switch workers to open a read-only DB connection and pull per-zone slices instead of inheriting the global frame (see §3.5). Validate RAM in the parent drops as expected.
5. **Drop municipality-level pickles.** Stop writing `{code}_{sbr,br}_results.pkl`. `read_br_inferred_indicators` / `read_sbr_inferred_indicators` become compatibility shims that query the DB. Add `br_handle()` / `sbr_handle()` siblings. `process_building_parts` returns `ResultsHandle`.
6. **`mergers.py` rewrite (the big one).** Change `mergers.py:796–897` to receive a `ResultsHandle` and do merge/groupby/dict construction in SQL. `mergers.merge()` returns `MunicipalityHandle`. This is the step that actually moves the **user-facing** RAM needle, because today's caller in `examples/buildings_complete_inference.py` holds the entire combined gdf to pickle it.
7. **External producer contract for `no_inference`.** Define the `no_inference` table schema in `_db.py` (so the external pipeline can target a real spec, not a pickle convention). Switch `read_addresses_indicators` to query that table. See §7.1.
8. **Examples sweep.** Update all 10 `examples/*.py` files listed in §2.2 to consume `MunicipalityHandle` and stop pickling. Helpers in `hc.utils` (`plot_building_reference`, `plot_building_shadows_analysis` per the example at line 74) gain a handle-aware overload.

Stop-points where a partial migration still leaves a working system:
- After step 3: per-zone pickles gone, but final pickles still produced. `read_*_indicators` still works as-is.
- After step 5: all package-internal pickles gone, downstream consumers (examples, external) still work via shims.
- After step 8: clean.

---

## 7. Decisions — all resolved

- Geometry → DuckDB spatial `GEOMETRY` (§3.3).
- `_no_inference.pkl` → external pipeline; we define the table and the loader, they target our schema (§7.1).
- Aux frames → migrated in this pass, with the worker-model change in §3.5 (step 4 in §6).
- **List/dict columns** → mix: `LIST<STRUCT>` for `building_spaces_*` (uniform inner schema), `JSON (VARCHAR)` for sbr_multi heterogeneous dicts (§3.4).
- **Resume semantics** → shared DB across runs. `zone_status` rows persist; a re-run with the same `(cadaster_code, results_dir)` skips zones marked `state = 'ok'`. To force a clean run, delete the DB file (mirrors today's "rm pkl" behaviour).
- **Examples sweep** → all 10 files in §2.2, in step 8 (§6).
- **DuckDB version pin** → `duckdb==1.5.3` in `requirements.txt` (matches what's already in your env from the prior WIP attempt; spatial extension is stable on this version).

All inputs resolved. Ready to start step 1.

### 7.1 `read_addresses_indicators` — same table, different projection

`_no_inference.pkl` is just `hc.merge(..., building_parts_inference=False)` written under a user-chosen filename. The schema is the 12 basic columns at `mergers.py:981–983`:

```
gml_id, building_status, building_reference, building_use,
building_geometry, year_of_construction, n_building_units, n_dwellings,
n_floors_above_ground, building_area, zone_reference, zone_type
```

These columns are a strict subset of what the inference path produces (the inference path adds br/sbr columns on top). So in the new model:

- One `buildings` table. When `building_parts_inference=False`, inference-only columns stay NULL.
- `read_addresses_indicators(code, wd)` becomes `SELECT <12 basic cols> FROM buildings WHERE cadaster_code = ?` — a projection, not a separate table.
- `MunicipalityHandle.addresses()` is just `self.buildings().select(*BASIC_COLS)`.
- No `no_inference` table. No external producer contract. Simpler than the original §7.1 sketch.

---

## 8. Risks

- **DuckDB single-writer**: only one process can write a DB at a time. Today's per-zone parallel workers each `to_pickle` independently. In the new model, **workers must not write to the central DB directly** — each worker writes its zone to a tiny per-zone DB file (`cache/{zone}.duckdb`), parent attaches and `INSERT INTO ... SELECT * FROM` to union. Same disk shape as today, no concurrency hazard. Step 2 in §6 implements this pattern.
- **Worker read-only contention on aux tables** (new with §3.5): workers all open `read_only=True` connections to the same DB to query `aux_footprints`. DuckDB supports concurrent readers, but if N workers each issue similar large queries, the OS page cache can get hammered. Mitigation: have each worker query *only* the slice it needs (`WHERE building_reference IN (...)`) and rely on DuckDB's predicate pushdown. If this proves slow, fall back to a per-shard-DB read pattern.
- **Geometry roundtrip cost**: even with the spatial extension, geometry conversion back to shapely (for shapely-using code) is not free. If the inner loop in `mergers.py` pulls geometry per-row, we can lose more than we save. Mitigation: keep geometry out of the hot SQL path; join on `building_reference` and materialise geometry only when emitting the final output (geojson/parquet).
- **Spatial extension portability**: the extension is downloaded on first `LOAD spatial;` from DuckDB's extension repo. If the user runs offline, this fails. Mitigation: document the requirement, optionally pre-bundle the extension. Confirm with user if offline runs matter.
- **Schema drift**: `br_results` column set evolves as the inference code grows. Need `ALTER TABLE ADD COLUMN IF NOT EXISTS` on first sight, or a JSON `extras` column for unknown fields. Recommendation: lazy column add — the `_db.py` writer introspects the DataFrame and adds missing columns before insert.
- **Stash recovery**: the prior WIP duckdb attempt (`stash@{0}`) may have hit some of these. Worth a 10-min skim before step 2 to avoid re-inventing fixes the user already debugged.
- **Stash recovery**: the prior WIP duckdb attempt (`stash@{0}`) may have already hit some of these. Worth a 10-min skim before step 2 to avoid re-inventing fixes.