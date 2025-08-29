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

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest flake8 black
```

## 🎯 Areas for Contribution


### **Performance Optimizations**
- Parallel processing improvements for large datasets
- Memory usage optimization for provincial-scale analysis
- Caching strategies for repeated analysis
- Database backend integration for large-scale deployments

### **Data Source Expansions**
- Integration of the Basque Country and Navarre cadasters
- Integration with additional Spanish open data sources
- Historical data integration capabilities
- Enhanced regional data source integration

### **Analysis Algorithm Improvements**
- Estimate dwellings distribution by each floor of the building
- Improved main orientation analysis (currently using parcel geometry)
- Integration with remote sensing data
- Enhanced shadow analysis and solar potential calculations

### **Visualization & Reporting**
- Web-based dashboard for analysis results
- Interactive visualization components
- Export capabilities to additional formats (KML, Shapefile, etc.)
- Automated report generation

### **External Tool Integration**
- Integration with building simulation tools for city-level energy simulations
- GIS software plugin development
- Building Information Modeling (BIM) format exports
- Integration with urban planning software

### **Documentation & Examples**
- Additional tutorial notebooks and scripts
- Video tutorials and documentation
- API reference documentation improvements
- Real-world case study examples
- Multi-language documentation

### **Testing & Quality Assurance**
- Comprehensive test suite expansion
- Performance benchmarking
- Data quality validation tools
- Continuous integration improvements

### **User Experience**
- Command-line interface improvements
- Configuration file support
- Progress monitoring enhancements
- Error message improvements

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