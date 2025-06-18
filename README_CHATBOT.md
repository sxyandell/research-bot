# 🧬 Genetic QTL Research Chatbot

A beautiful, interactive web chatbot for analyzing Quantitative Trait Loci (QTL) data with computational capabilities and AI-powered biological interpretation.

![Genetic Theme](https://img.shields.io/badge/Theme-Genetics-brightgreen)
![Flask](https://img.shields.io/badge/Flask-3.1.1-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![AI](https://img.shields.io/badge/AI-Gemini--1.5--Flash-purple)

## ✨ Features

### 🎨 Beautiful Genetics-Themed UI

- **DNA Double Helix Animations** - Floating DNA strands and nucleotide particles
- **Scientific Color Palette** - Adenine (Red), Thymine (Teal), Guanine (Blue), Cytosine (Green)
- **Responsive Design** - Works perfectly on desktop, tablet, and mobile
- **Modern Glass-morphism** - Blur effects and translucent surfaces
- **Interactive Elements** - Hover effects, smooth animations, typing indicators

### 🧮 Computational Capabilities

- **Statistical Analysis** - Calculate averages, counts, ranges from QTL data
- **Gene Discovery** - Find QTLs by highest/lowest LOD scores
- **Chromosome Analysis** - Analyze specific chromosomes
- **Regulation Types** - Compare cis-acting vs trans-acting QTLs
- **Real-time Computation** - Instant calculations from 500 QTL dataset

### 🤖 AI-Powered Interpretation

- **Biological Context** - Explains QTL significance and implications
- **Research Insights** - Provides liver biology and genetic interpretation
- **Smart Responses** - Combines computational results with expert knowledge
- **Natural Language** - Ask questions in plain English

### 🔬 Research Features

- **500 QTLs** from DO1200 liver tissue analysis
- **303 Unique Genes** across 19 chromosomes
- **RAG Integration** - Vector database for semantic search
- **ChromaDB** - Persistent embeddings for fast retrieval
- **Live Statistics** - Real-time dataset metrics in header

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Ensure you have Python 3.10+ installed
python3 --version

# Install required packages
pip install flask google-generativeai chromadb python-dotenv pandas
```

### 2. Setup Environment

```bash
# Create config.env file with your Google API key
echo "GOOGLE_API_KEY=your_google_api_key_here" > config.env
```

### 3. Generate QTL Data (if not already done)

```bash
# Generate the enhanced QTL chunks
python3 chunking.py

# Create vector database
python3 vectordb.py
```

### 4. Launch Chatbot

```bash
# Start on default port 5000
./start_chatbot.sh

# Or specify custom port
./start_chatbot.sh 8080

# Or run directly
python3 web_chatbot.py
```

### 5. Access the Chatbot

Open your browser and navigate to:

- **Local**: http://localhost:5000
- **Server**: http://your-server-ip:5000

## 🎯 Sample Queries

Try these example questions to explore the chatbot's capabilities:

### 📊 Statistical Queries

- "What is the average LOD score?"
- "How many QTLs are in the dataset?"
- "What's the range of LOD scores?"

### 🧬 Gene Discovery

- "Which gene has the highest LOD score?"
- "Show me the lowest LOD score QTL"
- "Find QTLs with LOD scores above 500"

### 🧭 Chromosome Analysis

- "Tell me about chromosome 11 QTLs"
- "How many QTLs are on chromosome 1?"
- "Compare chromosomes by QTL count"

### 🔄 Regulation Analysis

- "How many cis-acting vs trans-acting QTLs?"
- "What percentage are cis-acting?"
- "Explain cis vs trans regulation"

### 🧠 Biological Interpretation

- "What do high LOD scores mean?"
- "Explain the biological significance of these QTLs"
- "How do these QTLs affect liver function?"

## 🏗️ Project Structure

```
research-bot/
├── web_chatbot.py              # Main Flask application
├── templates/
│   └── index.html              # Beautiful genetics-themed UI
├── static/
│   ├── css/
│   │   └── style.css          # Comprehensive styling with animations
│   └── js/
│       └── chat.js            # Interactive chat functionality
├── enhanced_rag_chunks.json    # QTL data with biological context
├── enhanced_vectordb_chunks.json  # Streamlined chunks for vector DB
├── chroma_db/                  # ChromaDB persistent storage
├── chunking.py                 # Data preparation script
├── vectordb.py                 # Vector database creation
├── start_chatbot.sh           # Easy startup script
└── config.env                 # Environment variables
```

## 🔧 Configuration

### Flask Settings

The chatbot runs with these default settings:

- **Host**: 0.0.0.0 (accessible from any IP)
- **Port**: 5000 (configurable)
- **Debug**: True (disable for production)

### AI Model Configuration

- **Model**: Gemini-1.5-Flash (fast and efficient)
- **Embeddings**: Google embedding-001
- **Vector DB**: ChromaDB with persistent storage
- **Context**: Top 3 relevant chunks per query

### Computational Settings

- **Dataset**: 500 top QTLs by LOD score
- **Chunk Size**: 25 QTLs per chunk
- **Biological Enhancement**: Chromosome context + liver biology
- **Statistics**: Pre-computed for fast access

## 🚀 Deployment

### Local Development

```bash
# Run in development mode
python3 web_chatbot.py
```

### Production Server

```bash
# Install production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_chatbot:app

# Or with the startup script
./start_chatbot.sh 5000
```

### Server Configuration

For production deployment:

1. **Reverse Proxy** (nginx recommended):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

2. **SSL Certificate** (Let's Encrypt recommended)
3. **Environment Variables** in production config
4. **Firewall Rules** for port access

## 🎨 Customization

### Themes

The genetics theme uses these color variables in `style.css`:

```css
:root {
  --dna-a: #ff6b6b; /* Adenine - Red */
  --dna-t: #4ecdc4; /* Thymine - Teal */
  --dna-g: #45b7d1; /* Guanine - Blue */
  --dna-c: #96ceb4; /* Cytosine - Green */
}
```

### Animations

- DNA strand rotation: 20s duration
- Floating particles: 15s float animation
- Message transitions: 0.3s ease-out
- Typing indicators: 1.4s pulse animation

### Sample Queries

Add new sample queries in `templates/index.html`:

```html
<button class="query-button" onclick="sendSampleQuery('Your custom query')">
  Custom Query Button
</button>
```

## 🔍 Troubleshooting

### Common Issues

**Port Already in Use**

```bash
# Kill process using port 5000
sudo lsof -ti:5000 | xargs kill -9

# Or use different port
./start_chatbot.sh 5001
```

**Missing ChromaDB**

```bash
# Recreate vector database
python3 vectordb.py
```

**API Key Issues**

```bash
# Verify config.env file
cat config.env
# Should show: GOOGLE_API_KEY=your_key_here
```

**Missing Dependencies**

```bash
# Install all requirements
pip install flask google-generativeai chromadb python-dotenv pandas
```

## 📊 Performance

### Response Times

- **Computational Queries**: < 100ms (pre-computed stats)
- **AI Responses**: 1-3 seconds (Gemini API)
- **Vector Search**: < 200ms (ChromaDB)
- **UI Animations**: 60fps smooth

### Resource Usage

- **Memory**: ~200MB (including vector embeddings)
- **Storage**: ~50MB (vector database + chunks)
- **CPU**: Low (except during AI generation)

## 🎯 Future Enhancements

### Planned Features

- 📈 **Interactive Charts** - LOD score distributions, chromosome maps
- 🔍 **Advanced Filters** - Filter by gene type, chromosome, LOD range
- 📋 **Export Functionality** - Download results as CSV/JSON
- 👥 **Multi-user Support** - Session management and user preferences
- 🔔 **Real-time Updates** - WebSocket integration for live data
- 🧬 **3D Visualizations** - Interactive chromosome and gene models

### Possible Integrations

- **Gene Ontology** - Functional annotations
- **KEGG Pathways** - Metabolic pathway analysis
- **PubMed Search** - Related research papers
- **Protein Databases** - UniProt integration

## 📄 License

This project is part of genetic research at the University of Wisconsin-Madison. Please cite appropriately if used in research.

## 🤝 Contributing

For improvements or bug reports, please contact the research team.

## 📞 Support

For technical support or questions about the QTL data:

- Research Team: University of Wisconsin-Madison
- Server: attie.diabetes.wisc.edu
- Dataset: DO1200 Liver QTL Analysis

---

**Happy Genetic Research! 🧬🔬**
