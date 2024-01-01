import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

abs_file_path = "/home/prashant/Downloads/llm-detect-ai-generated-text/train_essays.csv"
train_data = pd.read_csv(abs_file_path)

print(train_data.head())
print("-----")

# Display the columns
print(train_data.columns)
print("-----")

# Check for missing values
print(train_data.isnull().sum())
print("-----")

# id_prompt_counts = train_data['id_prompt'].value_counts()
# print(id_prompt_counts)

sns.countplot(x='id', data=train_data)
plt.title('Distribution of Labels')
plt.show()

# Explore the text data
# Display a few examples of the 'text' column
print(train_data['text'].head())

# Explore the length of the text
train_data['text_length'] = train_data['text'].apply(len)
sns.histplot(train_data['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Length')
plt.show()
