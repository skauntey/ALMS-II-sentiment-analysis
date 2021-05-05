Readme Note
Step:1
Make sure that you first setup Kaggle API access before running the *. ipynb files. Model files depend on external data sources which won’t work without this.
The instructions to do so is available under ‘API Credential’ part of the Kaggle API documentation.
https://github.com/Kaggle/kaggle-api 
In simple steps,
1.	Log-in to Kaggle (or sign up)
2.	Navigate to your Account page (click top-right profile picture)
3.	Scroll down to the API section and click Create New API Token
4.	Save kaggle.json to the path as follow “C:\Users\<Windows-username>\.kaggle\kaggle.json”
Step 2:
Clone the repository: https://github.com/skauntey/ALMS-II-sentiment-analysis.git
This should work on both Colabs and Local drive.
Step3: Run models in Colabs or on the local machine. Make sure that you clone the repo first. 
File1: Conventional machine learning algorithms for text classification
https://github.com/skauntey/ALMS-II-sentiment-analysis/blob/main/skl_AMLSII_movie_reviews.ipynb
File 2: RNNs (LSTM, GRU, Bidirectional LSTM) for the text classification
https://github.com/skauntey/ALMS-II-sentiment-analysis/blob/main/rnn_ALMSII_movie_reviews.ipynb
