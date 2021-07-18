from tkinter import *
from tkinter.ttk import Combobox
import numpy as np
import Backprobagtion
root = Tk()
root.title("Back Propagation :")
root.geometry('1000x400')


Nummber_Of_hidden= Label(root, text="number Of Layers:")
Nummber_Of_hidden.place(x=100, y=100)
Nummber_Of_hiddentext = Entry(root)
Nummber_Of_hiddentext.place(x=200, y=100)
Nummber_Of_hiddentext.focus_set()

Number_Of_Neurons = Label(root, text="Num Of Neurons:")
Number_Of_Neurons.place(x=400, y=100)
Number_Of_Neurons_text = Entry(root)
Number_Of_Neurons_text.place(x=520, y=100)
Number_Of_Neurons_text.focus_set()

Learning_Rate = Label(root, text="Learning Rate")
Learning_Rate.place(x=100, y=200)
Learning_Rate_text = Entry(root)
Learning_Rate_text.place(x=200, y=200)
Learning_Rate_text.focus_set()

Epochs = Label(root, text="Epochs")
Epochs.place(x=400, y=200)
Epochs_text = Entry(root)
Epochs_text.place(x=500, y=200)
Epochs_text.focus_set()

Bias = Label(root , text = "Bias"  )
Bias.place(x = 650 , y =200)
Bias_text = Checkbutton(root , text="add bias?")
Bias_text.place(x=700 , y =200)

Activation_Function = Label(root, text="Activation_Function :")
Activation_Function.place(x=100, y=250)
V = ["Sigmoid", "Hyperbolic Tangent Sigmoid"]
Activation_Function_Text = Combobox(root, width=40, values=V)
Activation_Function_Text.place(x=250, y=250)


def RUN():


    list_layer = Number_Of_Neurons_text.get()
    learing_rate = float(Learning_Rate_text.get())
    epoch = int(Epochs_text.get())
    if Bias_text:
        bias = 1
    else:
        bias=0
    type = Activation_Function_Text.get()
    Backprobagtion.Run(list_layer,epoch,bias,learing_rate,type)
Run = Button(root, text="RUN", width=20,  command=RUN)
Run.place(x=400, y=300)


mainloop()
mainloop()