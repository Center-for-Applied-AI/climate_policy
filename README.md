# Climate Policy Analysis

This repository contains code and data for analyzing public comments on climate-related financial risk management policy proposals by U.S. banking regulators (Federal Reserve, OCC, and FDIC). The project examines how public feedback influenced the final policy text and explores the role of different stakeholder perspectives in the regulatory process.

## Data

The full dataset can be downloaded from:

https://www.dropbox.com/scl/fi/j5y6c8k2gpqxu90a7byod/climate_policy_data.zip?rlkey=d3y37xxgty2cgcbxjkunskapn&st=dwhs1j8j&dl=0

### Data Folder Structure

```
data/
├── comments/                    # Public comment letters submitted to regulators
│   ├── fed/
│   │   ├── pdf/                # Original PDF files (141 comments)
│   │   └── txt/                # Extracted text files
│   ├── fdic/
│   │   ├── pdf/                # Original PDF files (72 comments)
│   │   └── txt/                # Extracted text files
│   └── occ/
│       ├── pdf/                # Original PDF files (55 comments)
│       └── txt/                # Extracted text files
├── policies/
│   ├── drafts/                 # Initial policy proposals by each agency
│   │   ├── fed.txt
│   │   ├── fdic.txt
│   │   └── occ.txt
│   ├── final.txt               # Final joint policy text
│   └── parsed/                 # Parsed policy documents in various formats
└── temp/                       # Temporary processing files
```

## Scripts

### Data Processing

**s01_pdf_page_counter_and_parser.py** - Extracts text from PDF comment files using PyMuPDF, counts pages, detects overlapping content between comments using 25-word n-grams, and filters overlaps against policy markdown files. Saves parsed text to individual files and generates overlap reports.

**s02_count_text_characters.py** - Counts total characters in extracted comment text files across directories to assess corpus size and comment length distribution.

**s12_split_file_65.py** - Splits a specific FDIC comment file (c-065) that contains multiple sub-comments by a delimiter string, creating separate files for each sub-comment.

**s14_upd_occ.py** - Downloads OCC comment PDFs from URLs in metadata CSV, parses them using pdfplumber, extracts text, and generates summary statistics (file sizes, character counts).

**s19_extract_id_and_comment.py** - Extracts comment IDs and content from OCC and FDIC comments, cleans text by removing in-sentence newlines while preserving paragraph breaks, and generates GPT-powered summaries (up to 200 words per comment).

**s24_split_foia_comments.py** - Splits large FOIA PDF files containing multiple Federal Reserve comments into individual per-comment PDF files based on page ranges specified in an Excel file. Handles two-volume FOIA responses.

**s25_parse_fed_comments.py** - Parses Federal Reserve comment PDFs into text using the Marker library (with GPU acceleration support). Attempts Python API first, falls back to CLI if needed. Configurable for MPS/CUDA acceleration.

### Exploratory Analysis

**s03_describe_sections.py** - Computes cosine similarity matrices between draft policy paragraphs (FDIC, OCC, FED) and final policy paragraphs using OpenAI embeddings. Generates heatmaps with marginal bar plots showing paragraph lengths and saves detailed similarity tables to Excel.

**s04_paragraph_best_match.py** - For each draft policy paragraph, finds the single best-matching final policy paragraph based on cosine similarity. Also performs reverse matching (for each final paragraph, find best source). Saves match tables to Excel.

**s05_match_top3.py** - Extends s04 to find the top 3 best-matching paragraphs (instead of just 1) for each draft and final paragraph pair. Useful for identifying multiple potential influences.

**s06_match_top10.py** - Extends s04/s05 to find the top 10 best-matching paragraphs with word counts, providing broader context for paragraph-level changes.

**s07_paragraph_gpt_comparison.py** - Uses GPT-4 to qualitatively assess whether specific draft paragraphs influenced specific final paragraphs (top 3 matches from s05). Generates structured JSON outputs with influence assessments and explanations for paragraphs 65-83.

**s08_diff_vector_comment_similarity.py** - Computes embedding-based similarity between comments and the difference vector (final - draft) for each agency. Identifies comments most aligned with the direction of policy changes.

### Scoring and Classification

**s09_climate_score.py** - Uses GPT-4 with structured outputs (instructor library) to classify each comment on a 1-5 climate policy engagement scale (1=strong opposition, 5=strong advocate) and extracts author metadata (name, organization, location, type). Generates histogram of score distribution by agency.

**s10_plot_scores.py** - Generates publication-quality visualizations of climate engagement scores: individual histograms per agency, aggregated statistics by author type/organization/state with 95% confidence intervals. Outputs PNG, SVG, and TikZ formats for LaTeX.

**s22_gpt_paragraph_scoring.py** - Paragraph-level variant of s09 that scores individual paragraphs within comments for climate engagement (not used in main analysis).

**s23_analyze_streamlit.py** - Analyzes human survey data collected via Streamlit application comparing human and GPT assessments of comment influence.

### Similarity Analysis (Main Equations)

**s20_embedding_similarity_eq2.py** - **Equation 2**: Computes embedding-based cosine similarity between each comment and both draft and final policy paragraphs (65-83). For each paragraph, identifies the best-matching draft paragraph, then compares comment similarity to draft vs. final. Creates binary outcome variable (y_after_gt_before). Caches embeddings to JSON for efficiency.

**s21_regression_similarity_eq3.py** - **Equation 3**: Runs OLS and logit regressions relating comment climate engagement scores to whether comments are more similar to final vs. draft policy text (outcome from s20). Tests multiple specifications with agency controls, paragraph fixed effects, comment fixed effects, and various standard error clustering (HC1, paragraph-clustered, two-way clustered).

**s26_decile_alignment_analysis_eq4.py** - **Equation 4**: Analyzes the relationship between comment climate scores and the change in similarity (delta = sim_after - sim_before) from draft to final policy. Groups comments by score deciles and plots average delta. Runs regressions with paragraph-clustered standard errors.

### Topic Analysis

**s15_tfidf_topic_analysis.py** - Performs TF-IDF analysis on comment text to extract top 25 unigrams and bigrams by climate engagement score and by agency. Uses custom preprocessing to remove URLs, numbers, and stopwords. Generates bar plots and Excel tables.

**s16_topic_extraction.py** - Uses GPT-4 with structured outputs to extract up to 3 topics (max 3 words each) from each comment. Generates topic frequency statistics and plots. Converts topics to dummy variables for regression analysis.

**s17_ols_topics.py** - Runs OLS regressions with topic dummies as predictors of climate engagement scores. Merges semantically similar topics using embedding-based cosine similarity (threshold 0.9). Computes VIF to check multicollinearity.

**s18_topic_correlation.py** - Computes and visualizes correlation matrix of topic dummies after merging similar topics. Generates heatmap showing which topics co-occur in comments.

**s28_eq3_topics.py** - Augments Equation 3 regressions (from s21) by adding significant topic dummies (p<0.05 from s17) as additional controls. Tests whether topic controls affect the relationship between climate scores and policy influence.

**s34_accept_and_revise_topics.py** - Extracts topics from role-based revised policies (from s33), computes TF-IDF similarity between revised policies and official final policy, and analyzes distinctive topics by role. Generates topic frequency plots and role-specific topic analysis.

### Influence Assessment

**s11_gpt_influence_assessment.py** - Uses GPT-4 to assess whether specific comments influenced specific policy paragraph changes (paragraphs 65-83). For each (paragraph, comment) pair, GPT decides "yes" or "no" with explanation. Then runs regressions linking binary influence labels to comment engagement scores with multiple fixed effects and clustering specifications. Uses caching to avoid redundant API calls.

**s13_ols.py** - Runs OLS and logit regressions linking GPT-assessed influence (from s11) to comment climate engagement scores. Tests multiple specifications with agency controls, paragraph fixed effects, and various standard error types (HC1, paragraph-clustered, two-way clustered).

### Monte Carlo Simulations

**s27_frs_fed_revise.py** - Single-iteration role-based policy revision simulation. For each regulatory role (monetary, banking, bureaucrat, partisan, etc.), performs iterative batch revision of policy text using accepted comments. Computes cosine similarity between role-revised policies and official final policy. Generates similarity matrices and heatmaps.

**s29_frs_fed_revise_mc.py** - Monte Carlo version of s27. Runs 100+ iterations with different random seeds for each role, sampling comments in batches and performing iterative revisions. Implements rate limiting for GPT-5 API. Generates distributions of similarity scores with confidence intervals and ridge plots.

**s30_role_based_acceptance_revision.py** - Two-stage simulation: (1) For each role, uses GPT to decide which comments to accept (binary decision with brief summary of accepted changes), (2) Aggregates all accepted proposals and performs single-shot policy revision. Computes similarity to official final policy and generates acceptance rate plots by role. Includes plan-only mode for token estimation.

**s31_accept_and_revise_mc.py** - Monte Carlo version of s30 using GPT-5. Runs multiple iterations with random comment sampling to estimate acceptance rates and policy similarity distributions with confidence intervals. Tests 12 different roles including monetary, banking, bureaucrat, partisan, wealth-focused, and control roles (French cheese, Cinderella, OpenAI training). Generates stacked bar plots, t-test matrices comparing roles, and comprehensive similarity analysis.

**s32_role_rankings.py** - Post-processes s31 outputs to rank regulatory roles by their similarity to the official final policy. Uses bootstrap resampling (10,000 iterations) to estimate confidence intervals for mean ranks. Generates summary tables and plots showing which roles best predict actual policy outcomes.

**s33_accept_and_revise_mc.py** - **Gemini version** of s31. Uses Google's Gemini 2.5 Pro model instead of GPT-5 for comparison. Implements rate limiting and uses Gemini's embedding model. Tests same 12 roles to compare model behavior in role-based policy revision tasks.

## Dependencies

All dependencies are specified in `pyproject.toml`. Key packages include:

- `openai` - GPT API access for classification and analysis
- `instructor` - Structured outputs from LLMs
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - TF-IDF and machine learning
- `statsmodels` - Regression analysis
- `matplotlib`, `seaborn` - Visualization
- `PyMuPDF`, `pdfplumber` - PDF parsing
- `tqdm` - Progress bars

Install with:
```bash
pip install -e .
```

## Configuration

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"  # Required only for s33 (Gemini comparison)
```

## Outputs

Results are saved to:
- `outputs/` - CSV, Excel, and JSON files with analysis results
- `plots/` - PNG, SVG, and TikZ visualizations