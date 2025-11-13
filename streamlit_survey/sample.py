import pandas as pd
df = pd.read_csv("comments_id_and_content_full.csv")
df = df.sample(156, random_state=42)
df.to_csv("comments_id_and_content_survey.csv")
