# LLM Research Tool for Biology

A comprehensive toolkit for leveraging Large Language Models in biological research, focusing on DNA/Protein sequence analysis and HLA variant prediction.

## Features

- DNA Language Model for sequence analysis
- Protein Language Model for structure prediction
- HLA sequence variant analysis for transplant outcomes
- RESTful API for model inference
- Web interface for visualization and analysis

## Project Structure

```
llm_research_tool/
├── api/                 # API endpoints and routes
├── models/             # ML model implementations
├── data/               # Data processing and management
├── utils/              # Utility functions
├── tests/              # Test suite
├── config/             # Configuration files
├── docs/               # Documentation
└── web/                # Web interface
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Development

- Run tests: `pytest`
- Start API server: `python -m api.main`
- Start web interface: `cd web && npm start`

## License

MIT License

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
