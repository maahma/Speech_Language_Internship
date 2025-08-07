import nltk
import re
import string
import oscd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# MAKE LOOP TO GO THROUGH ALL THE CONFERENCE CALLS TEXT TRANSCRIPTS
# STORE CLEANED TEXT IN FOLDERS FOR EACH CONFERENCE CALL

class CleanText:
    def __init__(self, text):
        self.text = text

    # def split_into_sentences(self, text):
    #     sentences = text.split('\n')
    #     sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    #     return sentences
    
    def split_into_tokens(self, text):
        tokens = word_tokenize(text)
        return tokens

    def to_lowercase(self, tokens):
        tokens = [token.lower() for token in tokens]
        return tokens

    # def stem_words(self, tokens):
    #     porter = PorterStemmer()
    #     stemmed = [porter.stem(word) for word in tokens]
    #     print(stemmed[:100])
    #     return stemmed

    def filter_punctuation(self, tokens):
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        return words

    def filter_stopwords(self, words):
        stop_words = stopwords.words("english")
        words = [word for word in words if not word in stop_words]
        return words

    def clean(self):
        tokens = self.split_into_tokens(self.text)
        lowercase_words = self.to_lowercase(tokens)
        # file_stemmed = self.stem_words(file_lowercase)
        no_punctuation = self.filter_punctuation(lowercase_words)
        clean_words = self.filter_stopwords(no_punctuation)
        return clean_words

def process_transcripts(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Clean the text
                cleaner = CleanText(text)
                cleaned_text = cleaner.clean()

                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                output_file = os.path.join(output_subdir, file)
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(" ".join(cleaned_text))
                print(f"Processed and saved: {output_file}")

if __name__ == "__main__":
    input_dir = "./Recordings/ACL19_Release"
    output_dir = "./Cleaned_Transcripts"
    process_transcripts(input_dir, output_dir)