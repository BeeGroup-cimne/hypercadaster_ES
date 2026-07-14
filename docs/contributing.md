# Contributing to hypercadaster_ES

We welcome and encourage contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing usage examples, your contributions help make hypercadaster_ES better for everyone.

## 🤝 How to Contribute

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/hypercadaster_ES.git
cd hypercadaster_ES

# Add upstream remote for staying current with main repository
git remote add upstream https://github.com/BeeGroup-cimne/hypercadaster_ES.git
```

### Development Setup

```bash
# Create development environment
python -m venv hypercadaster_dev
source hypercadaster_dev/bin/activate  # Linux/Mac
# or: hypercadaster_dev\Scripts\activate  # Windows

# Install in development mode with the dev tools (pytest, black, ruff, pdoc)
pip install -e ".[dev]"
```

## 🧪 Running the Tests

The suite lives in [`tests/`](../tests) and is split into a fast, offline core
and one network-gated end-to-end smoke test.

```bash
# Fast, offline unit tests (no downloads) — run these before every PR
pytest

# Include the live smoke test (downloads a tiny municipality, ~2 min)
HC_RUN_LIVE=1 pytest      # or: pytest -m live
```

The offline tests cover the code conversions, grid/H3 aggregation, the energy
footprint geometry helpers, weather statistics/EPW export, and the public
import surface. Please add or extend tests alongside any behavioural change;
keep new tests offline and deterministic unless they genuinely require the
network (mark those with `@pytest.mark.live`).

## 📚 Building the API Reference

The HTML API reference is generated from the docstrings with
[`pdoc`](https://pdoc.dev) (Unlicense — permissive, matching our
permissive-dependency policy):

```bash
./scripts/build_api_docs.sh      # writes docs/api/index.html (git-ignored)
```

Keep public functions and classes documented with clear docstrings so the
generated reference stays useful.

## 🎯 Areas for Contribution

- Integrate the Basque Country and Navarre cadasters
- Web-based dashboard for analysis results
- Interoperability with VoxCity and city2graph
- Integrate a serialised model for predicting WWR, façade material composition and energy systems based on the `epc_CAT` library

> `social_ES` INE indicator integration (income atlas, population, education,
> rental & housing prices, electricity, Essential Characteristics) shipped in
> v1.2.0 — see `hypercadaster_ES/social.py` and `merge(social_ES_layer=True)`.

## 🛠️ Development Guidelines

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add feature: description of changes"

# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## 📝 Contribution Types

### Bug Fixes

1. **Identify the Issue**: Check existing issues or create a new one
2. **Reproduce the Bug**: Create a minimal example
3. **Fix the Issue**: Implement and test the fix
4. **Submit Pull Request**: Include test case and description

### New Features

1. **Discuss First**: Open an issue to discuss the feature
2. **Design**: Plan the implementation approach
3. **Implement**: Write code following our guidelines
4. **Test**: Add comprehensive tests
5. **Document**: Update documentation
6. **Submit**: Create pull request with detailed description

### Documentation Improvements

1. **Identify Gaps**: Find areas needing better documentation
2. **Write Content**: Create clear, helpful documentation
3. **Test Examples**: Ensure code examples work
4. **Submit**: Pull request with documentation updates

### Performance Improvements

1. **Profile Code**: Identify performance bottlenecks
2. **Benchmark**: Create performance benchmarks
3. **Optimize**: Implement improvements
4. **Validate**: Ensure optimizations don't break functionality
5. **Document**: Explain performance improvements
Thank you for considering contributing to hypercadaster_ES! Every contribution, no matter how small, helps improve the library for the entire community.

---

[← Output Data Schema](output-schema.md) | [Changelog →](changelog.md) | [Back to README](../README.md)