import random

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import telebot
bot = telebot.TeleBot('')

tok = GPT2Tokenizer.from_pretrained("/home/alina/Загрузки/checkpoint-5500")
model = GPT2LMHeadModel.from_pretrained("/home/alina/Загрузки/checkpoint-5500")

length = 0
start = ""


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, Владимир Владимирович на связи! Давай что-нибудь сочиним!")
        bot.register_next_step_handler(message, next_step)


def next_step(message):
    bot.send_message(message.from_user.id, "Введи желаемую длину")
    bot.register_next_step_handler(message, get_length)


def get_length(message):
    global length
    length = int(message.text)
    bot.send_message(message.from_user.id, "Введи начало произведения")
    bot.register_next_step_handler(message, get_begin)


def get_begin(message):
    global start
    start = message.text
    bot.send_message(message.from_user.id, "Ожидаем...")
    inpt = tok.encode(start, return_tensors="pt")
    k = random.randrange(1, 20)
    t = float(random.randrange(1, 50))
    out = model.generate(inpt, max_length=length, repetition_penalty=3.0,
                         do_sample=True, top_k=k, top_p=0.95, temperature=t)
    answer = tok.decode(out[0])
    bot.send_message(message.from_user.id, answer)
bot.polling()
