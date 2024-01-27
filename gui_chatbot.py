import tkinter
import pickle
import random
import json
import numpy as np

from tkinter import *
from keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print('found in bag: %s' % word)
    return np.array(bag)


def predict_class(sentence):
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, 'You: ' + msg + '\n\n')
        ChatLog.config(foreground='#442265', font=('Verdana', 12))

        ints = predict_class(msg)
        res = get_response(ints, intents)
        ChatLog.insert(END, 'Bot: ' + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


root = Tk()
root.title('Chatbot')
root.geometry('400x500')
root.resizable(width=FALSE, height=FALSE)

ChatLog = Text(root, bd=0, bg='white', height='8', width='50', font='Arial')
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(root, command=ChatLog.yview, cursor='heart')
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(root, font=('Verdana', 12, 'bold'), text='Send', width='12', height=5,
                    bd=0, bg='#32de97', activebackground='#3c9d9b', fg='#ffffff',
                    command=send)

EntryBox = Text(root, bd=0, bg='white', width='29', height='5', font='Arial')

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()
