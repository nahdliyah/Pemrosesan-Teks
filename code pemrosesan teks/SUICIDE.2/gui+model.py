from tkinter import *
from tkinter import Tk, Text, Button, Label, filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PIL import Image
import pytesseract as tess
tess.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class IntroLogin:
    def __init__(self, master):
        self.master = master
        master.geometry('1280x800+50+40')
        master.resizable(False, False)
        master.overrideredirect(True)
        master.config(bg='#3d3d3d')
        self.loginPage_pad = None
        self.loginInterface()

    def loginInterface(self):
        loginPage = Image.open('Login.jpeg')
        loginpage_resize = loginPage.resize((1280, 800))
        loginPage = ImageTk.PhotoImage(loginpage_resize)
        self.loginPage_pad = Label(image=loginPage, text='haha')
        self.loginPage_pad.image = loginPage
        self.loginPage_pad.pack()

        learnmoreButton = Image.open('LearnMore.jpeg')
        learnmoreButton_resize = learnmoreButton.resize((175, 40))
        learnmoreButton = ImageTk.PhotoImage(learnmoreButton_resize)
        learnmoreButton_pad = Button(image=learnmoreButton, borderwidth=0, highlightthickness=0,
                                     bd=0, command=self.tampilkanHalamanKedua, activebackground='grey',
                                     cursor='hand2')
        learnmoreButton_pad.image = learnmoreButton
        learnmoreButton_pad.place(x=230, y=550)

        moveBar = Image.open('movebar.jpeg')
        moveBar_resize = moveBar.resize((40, 40))
        moveBar = ImageTk.PhotoImage(moveBar_resize)
        moveBar_pad = Label(image=moveBar, bd=0)
        moveBar_pad.image = moveBar
        moveBar_pad.place(x=58, y=10)
        moveBar_pad.bind('<B1-Motion>', self.moveApp)

        closeLogo = Image.open('close.jpeg')
        closeLogo_resize = closeLogo.resize((40, 40))
        closeLogo = ImageTk.PhotoImage(closeLogo_resize)
        closePad = Button(image=closeLogo, borderwidth=0, highlightthickness=0, bd=0,
                          command=self.closeProgram, activebackground='grey')
        closePad.image = closeLogo
        closePad.place(x=10, y=10)

    def moveApp(self, e):
        root.geometry(f'+{(e.x_root)-58}+{(e.y_root)-10}')

    def closeProgram(self):
        root.quit()

    def tampilkanHalamanKedua(self):
        self.master.destroy()
        HalamanKedua()

class HalamanKedua:
    def __init__(self):
        self.root = Tk()
        self.root.geometry('1280x800+50+40')
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        self.root.config(bg='#3d3d3d')
        self.tampilkanHalamanKedua()
    
    def closeProgram(self):
        root.quit()

    def tampilkanHalamanKedua(self):
        gambarHalamanKedua = Image.open('bag2fix.jpeg')
        gambarHalamanKedua_resize = gambarHalamanKedua.resize((1280, 800))
        gambarHalamanKedua = ImageTk.PhotoImage(gambarHalamanKedua_resize)
        padGambarHalamanKedua = Label(self.root, image=gambarHalamanKedua, compound='center',
                                      font=('Helvetica', 20), fg='white', bg='#3d3d3d')
        padGambarHalamanKedua.image = gambarHalamanKedua
        padGambarHalamanKedua.pack()

        closeLogo = Image.open('close.jpeg')
        closeLogo_resize = closeLogo.resize((40, 40))
        closeLogo = ImageTk.PhotoImage(closeLogo_resize)
        closePad = Button(image=closeLogo, borderwidth=0, highlightthickness=0, bd=0,
                          command=self.closeProgram, activebackground='grey')
        closePad.image = closeLogo
        closePad.place(x=10, y=10)

        frame_isian = Frame(self.root, bd=0, width=700, height=280, bg='white')
        frame_isian.place(x=220, y=230)

        self.text_input = Text(frame_isian, font=('arial', 12), width=95, height=17, bg='white', fg='black', bd=0)
        self.text_input.pack(side=LEFT, fill=BOTH, padx=2)
        scrollbar = Scrollbar(frame_isian)
        scrollbar.pack(side=RIGHT, fill=BOTH)

        self.text_input.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_input.yview)

        # frame_isian = Frame(self.root, bd=0, width=700, height=280, bg='white')
        # frame_isian.place(x=220, y=230)
        
        # # Create a Text widget for user input
        # self.text_input = Text(frame_isian, font=('arial', 12), width=95, height=2, bg='white', fg='black', bd=0)
        # self.text_input.pack(side=LEFT, fill=BOTH, padx=2)

        

        # Create a button to trigger the prediction
        # predict_button = Button(frame_isian, text="Predict", command=self.predict_input)
        # predict_button.pack()
        
        submitButton = Image.open('submit.jpeg')
        submitButton_resize = submitButton.resize((175, 40))
        submitButton = ImageTk.PhotoImage(submitButton_resize)
        submitButton_pad = Button( image=submitButton, borderwidth=0, highlightthickness=0,
                                     bd=0,  command=self.predict_input, activebackground='black',
                                     cursor='hand2')
        submitButton_pad.image = submitButton
        submitButton_pad.place(x=784, y=634)
        # Create a button to delete the content of the Text widget
        # Create a button to delete the content of the Text widget
        deleteButtonImage = Image.open('delete.jpg')
        deleteButtonImage_resize = deleteButtonImage.resize((22, 22))
        deleteButtonImage = ImageTk.PhotoImage(deleteButtonImage_resize)
        deleteButton_pad = Button(image=deleteButtonImage, borderwidth=0, highlightthickness=0,
                                    bd=0,  command=self.delete_content, activebackground='black',
                                    cursor='hand2')
        deleteButton_pad.image = deleteButtonImage
        deleteButton_pad.place(x=1019, y=196)

        ocrButton = Image.open('OCR.jpg')
        ocrButton_resize = ocrButton.resize((175, 40))
        ocrButton = ImageTk.PhotoImage(ocrButton_resize)
        ocrButton_pad = Button( image=ocrButton, borderwidth=0, highlightthickness=0,
                                     bd=0,  command=self.openimage, activebackground='black',
                                     cursor='hand2')
        ocrButton_pad.image = ocrButton
        ocrButton_pad.place(x=1000, y=634)
        
        

        closeLogo = Image.open('close.jpeg')
        closeLogo_resize = closeLogo.resize((40, 40))
        closeLogo = ImageTk.PhotoImage(closeLogo_resize)
        closePad = Button(image=closeLogo, borderwidth=0, highlightthickness=0, bd=0,
                          command=self.closeProgram, activebackground='grey')
        closePad.image = closeLogo
        closePad.place(x=10, y=10)

    def openimage(self):
        filename = filedialog.askopenfilename()
        img1 = Image.open(filename)
        get_txt = tess.image_to_string(img1, lang='eng')
        print(get_txt)
        self.text_input.insert('0.0',get_txt)

    def delete_content(self):
        # Clear the content of the Text widget
        self.text_input.delete("1.0", END)
        
    

    def predict_input(self):
        # Get the input from the Text widget
        input_text = self.text_input.get("1.0", END)

        # Preprocess the input
        preprocessed_text = preprocess(input_text)

        # Count Vectorization
        input_vectorized = count_vectorizer.transform([preprocessed_text])

        # Predict
        prediction = modelLSBOW.predict(input_vectorized)

        # Display the prediction result in a message box
        messagebox.showinfo("Prediction Result", f"The prediction is: {prediction[0]}")

    # # Tentukan lokasi Tesseract OCR engine jika berbeda
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # def open_file_dialog():
        
    #     # Buka file explorer untuk memilih gambar
    #     file_path = filedialog.askopenfilename()
    #     if file_path:
    #         extract_text(file_path)

    # def extract_text(image_path):
    #     # Buka gambar menggunakan PIL
    #     img = Image.open(image_path)
        
    #     # Lakukan OCR pada gambar
    #     text = pytesseract.image_to_string(img)
        
    #     # Tampilkan hasilnya
    #     print(text)

    # # Gunakan fungsi untuk memilih gambar
    # open_file_dialog()
        
    

import pandas as pd
# from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
# PCA, TruncatedSVD
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


df = pd.read_csv('ds.csv')
# df

#=====================================================================================
# Membuang kolom kutipan yang berisi string kosong []
df = df.loc[df['Tweet'] != '[]']

count_columns_with_empty_string = (df == '[]').sum()
# print(count_columns_with_empty_string)

id_stopword_dict = pd.read_csv('ST.csv', header=None, encoding='latin-1')
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})
# id_stopword_dict.head()

#=====================================================================================
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "coz" : "because",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "didn" : "didn't",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "eng":"english",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "ftw" : "for the win",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "ger" : "German",
    "ger/rus/eng" : "german,rusian,english",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "hwo're":"how are",
    "hav" : "have",
    "hoe" : "whore",
    "herme" : "hear me",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "ill" : "i'll",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "jst": "just",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    'pls' : "please",
    "plz" : "please",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "rus" :"rusian",
    "re" : "are",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "tf" : "the fuck",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wot" : "what",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "wit" : "with",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "youve" : "you've",
    "yr" : "your",
    "ve" : "have",
    "zzz" : "sleeping bored and tired"
}

def convert_abbrev(word):
    if ',' in word:
        # Jika ada koma, pisahkan kata dan cek kamus
        words = word.split(',')
        return ', '.join(abbreviations.get(w.lower(), w) for w in words)
    # elif word.startswith('@'):
    #     # Memisahkan kata dari kode seperti 'ger/rus/eng'
    #     code_word = word[1:]
        # return abbreviations.get(code_word, word)
    else:
        # Menggunakan kamus untuk mengganti kata
        return abbreviations.get(word.lower(), word)

# Mengaplikasikan fungsi pada DataFrame
df['preprocess'] = df['Tweet'].astype(str).apply(lambda x: ' '.join(convert_abbrev(word) for word in x.split()))

#===================================================================================================

import contractions
import pandas as pd
import html
# Fungsi untuk memperluas kontraksi
def expand_contractions(text):
    text = html.unescape(text)
    return contractions.fix(text)
df['preprocess'] = df['preprocess'].apply(expand_contractions)

import contractions
import pandas as pd

# Fungsi untuk memperluas kontraksi
def expand2_contractions(text):
    return contractions.fix(text)
df['preprocess'] = df['preprocess'].apply(expand2_contractions)
# df

#===============================================================================================================
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import html

nltk.download('punkt')

from nltk.stem import PorterStemmer

def lowercase(text):
    return text.lower()

punct = string.punctuation
def remove_punctuation(text):
    no_punct= [words for words in text if words not in punct]
    words_wo_punct = ''.join(no_punct)
    
    return words_wo_punct

def remove_unnecessary_char(text):
    text = re.sub('\n', ' ', text) # Remove every '\n'
    text = re.sub('user', ' ', text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text) # Remove every url
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = re.sub('@[\w_]+', ' ', text) # Remove @
    text = re.sub('#[\w_]+', ' ', text) # Remove tag hashtag
    text = re.sub('Ã¢Â€Â™', "'",text)
    text = re.sub('â', "",text)
    text = re.sub('\d+', '', text)

    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text

def remove_stopword(text):
    text = ' '.join([ ' ' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip() 
    return text

# import num2words

# def convert_number_to_words(number):
#     if isinstance(number, str) and number.isdigit():
#         number = int(number)
#     if isinstance(number, int):
#         return num2words.num2words(number)
#     else:
#         return number
    
def stemming(text):
    # Tokenisasi teks menjadi kata-kata
    words = word_tokenize(text)
    
    # Inisialisasi stemmer Porter
    porter = PorterStemmer()
    
    # Melakukan stemming pada setiap kata
    stemmed_words = [porter.stem(word) for word in words]
    
    # Menggabungkan kata-kata yang telah distem menjadi sebuah teks kembali
    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def english_lemmatization(text):
    # Tokenisasi teks menjadi kata-kata
    words = word_tokenize(text)
    
    # Inisialisasi lemmatizer WordNet
    lemmatizer = WordNetLemmatizer()
    
    # Melakukan lemmatization pada setiap kata
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Menggabungkan kata-kata yang telah dilemat menjadi sebuah teks kembali
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

from nltk.stem import LancasterStemmer

def lancaster_stemming(text):
    words = word_tokenize(text)
    lancaster = LancasterStemmer()
    stemmed_words = [lancaster.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

import contractions
import pandas as pd

# Fungsi untuk memperluas kontraksi
def expand2_contractions(text):
    return contractions.fix(text)

def convert_abbrev(text):
    return abbreviations[text.lower()] if text.lower() in abbreviations.keys() else text

    
# def double_word(text):
#     text = re.sub("(.)\\1{2,}","\\2",text)
#     return text

def preprocess(text):
    text = html.unescape(text)
    text= convert_abbrev(text)
    # text = expand2_contractions(text)
    # text= convert_number_to_words(text)
    text = lowercase (text) # 1
    text = remove_unnecessary_char(text) 
    text = remove_nonaplhanumeric(text) 
    text = expand2_contractions(text)
    text = remove_stopword (text)
    text= remove_punctuation(text)
    # text = stemming(text)
    text = english_lemmatization(text)
    # text= lancaster_stemming(text)
    
    # text = double_word(text)
    
    return text

df['preprocess'] = df['preprocess'].apply(preprocess)
# df.to_csv('clean.csv', index=False)
#===============================================================================================
import pandas as pd

data1 = pd.read_csv("clean.csv")
data1

data1['preprocess'].fillna('', inplace=True)
data1['Tweet'].fillna('', inplace=True)

list_corpus = data1["preprocess"].tolist()
list_labels = data1["Suicide"].tolist()


X_train, X_test, Y_train, Y_test = train_test_split(list_corpus, list_labels, test_size=0.2)

def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# def plot_LSA(test_data, test_labels, plot=True):
#         #      savepath="LSA_demo.csv", plot=True):
# #LSA digunakan untuk mengurangi dimensi fitur menjadi 2 dimensi. Parameter n_components=2 menunjukkan bahwa kita ingin mendapatkan dua komponen utama.
#     lsa = TruncatedSVD(n_components=2)
#     lsa_scores = lsa.fit_transform(test_data)
#     color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
#     color_column = [color_mapper[label] for label in test_labels]
#     colors = ['orange', 'blue']
#     if plot:
#         plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=color_column, cmap=matplotlib.colors.ListedColormap(colors))
#         orange_patch = mpatches.Patch(color='orange', label='Suicide')
#         blue_patch = mpatches.Patch(color='blue', label='Not suicide')
#         plt.legend(handles=[orange_patch, blue_patch], prop={'size': 10})

#         # # Simpan data ke dalam file CSV
#         # result_df = pd.DataFrame({'LSA1': lsa_scores[:, 0], 'LSA2': lsa_scores[:, 1], 'Label': test_labels})
#         # result_df.to_csv(savepath, index=False)

# fig = plt.figure(figsize=(6,6))
# plot_LSA(X_train_counts, Y_train)
#         #  savepath="LSA_demo.csv")
# # plt.show()
#====================================================================================================
from sklearn.svm import LinearSVC
# Inisialisasi model
modelLSBOW = LinearSVC()

# Latih model dengan data pelatihan
modelLSBOW.fit(X_train_counts, Y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import scikitplot as skplt 

# Prediksi menggunakan data pengujian
yy_predd = modelLSBOW.predict(X_test_counts)

# Evaluasi performa model
# print("Akurasi:", accuracy_score(Y_test, yy_predd))
# print("Laporan Klasifikasi:\n", classification_report(Y_test, yy_predd))
# print("Matriks confusion:\n", confusion_matrix(Y_test, yy_predd))
# skplt.metrics.plot_confusion_matrix(Y_test, yy_predd, normalize=False, title = 'Matrix Confusion',figsize=(5,3))

#===============================================================================================================


root = Tk()
SUICIDEIntro = IntroLogin(root)
root.mainloop()
