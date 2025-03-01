### AI Research Chatbot

- Dataset : [https://www.kaggle.com/datasets/Cornell-University/arxiv/code]

```bash
conda create -n faiss_env python=3.10 -y
```
```bash
conda activate faiss_env
```
```bash
pip install -r requirements.txt
```

- How to run

```bash
cd src
```

```bash
streamlit run app.py
```

- Folder structure

```plaintext

AI_Expert_Chatbot/
|--- faiss_db/
|--- |--- index.faiss
|--- |--- index.pkl
|--- Notebook/
|--- |--- creating_Vector_DB_FAISS.ipynb # google colab notebook for creating vector db
|--- |--- Experiment1.ipynb # local notebook for creating vector db
|--- |--- Experiment2.ipynb # chat bot testing 
|--- Report/
|--- |--- Report.pdf
|--- src/
|--- |--- helper.py
|--- |--- app.py
|--- .env
|--- demo.json 
|--- README.md
|--- requirements.txt

```

