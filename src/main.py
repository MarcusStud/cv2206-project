import cv2
from v1 import v1main as v1
from v2 import v2main as v2
import PySimpleGUI as pg
import tkinter
import PIL
import re

def UpdateOutput(test) :
    for i in range (2):
        for j in range (4):
            test[i][j] = re.sub('[,()\']', '', str(test[i][j]))
    window.FindElement('OUTPUT1').Update(test[0][0])
    window.FindElement('OUTPUT2').Update(test[0][1])
    window.FindElement('OUTPUT3').Update(test[0][2])
    window.FindElement('OUTPUT4').Update(test[0][3])
    window.FindElement('OUTPUT5').Update(test[1][0])
    window.FindElement('OUTPUT6').Update(test[1][1])
    window.FindElement('OUTPUT7').Update(test[1][2])
    window.FindElement('OUTPUT8').Update(test[1][3])

def clearWindow() : 
    window.FindElement('OUTPUT1').Update("")
    window.FindElement('OUTPUT2').Update("")
    window.FindElement('OUTPUT3').Update("")
    window.FindElement('OUTPUT4').Update("")
    window.FindElement('OUTPUT5').Update("")
    window.FindElement('OUTPUT6').Update("")
    window.FindElement('OUTPUT7').Update("")
    window.FindElement('OUTPUT8').Update("")

def OutputSingleLine(tx):
    window.FindElement('OUTPUT1').Update(tx)

pg.theme("DarkAmber")
layout = [
    [pg.Text("How many pictures do you want to test?")],
    [pg.InputText()],
    [pg.Button("Check with version 1"), pg.Button("Check with version 2"), pg.Button("Clear")],
    [pg.Button("Select image version 1"), pg.Button("Select image version 2")],
    [pg.Text('', key = 'OUTPUT1')],
    [pg.Text('', key = 'OUTPUT2')],
    [pg.Text('', key = 'OUTPUT3')],
    [pg.Text('', key = 'OUTPUT4')],
    [pg.Text('', key = 'OUTPUT5')],
    [pg.Text('', key = 'OUTPUT6')],
    [pg.Text('', key = 'OUTPUT7')],
    [pg.Text('', key = 'OUTPUT8')],
]
window = pg.Window("Crack", layout)
while True: 
    event, values = window.read()
    if event == pg.WIN_CLOSED:
        break
    if event == "Clear" or event == pg.WIN_CLOSED:
        clearWindow()
    if event == "Check with version 1":
        if values[0] != "":
            res = v1.v1main(int(values[0]))
            UpdateOutput(res)
        else: 
            res = v1.v1main()
            UpdateOutput(res)
    if event == "Check with version 2":
        if values[0] != "":
            res = v2.v2main(int(values[0]))
            UpdateOutput(res)
        else: 
            res = v2.v2main()
            UpdateOutput(res)
    if event == "Select image version 1":
        clearWindow()
        path = tkinter.filedialog.askopenfilename(title="Select an Image", filetype=(('image    files','*.jpg'),('all files','*.*')))
        if path != "":
            img = PIL.Image.open(path)
            res = v1.functions.checkCrackSelectedImage(img)
            if res:
                OutputSingleLine("The image contains a crack")
            else:
                OutputSingleLine("The image does not contain a crack")
    if event == "Select image version 2":
        clearWindow()
        path= tkinter.filedialog.askopenfilename(title="Select an Image", filetype=(('image    files','*.jpg'),('all files','*.*')))
        if path != "":
            img = cv2.imread(path)
            res = v2.functions.checkCrackSelectedImage(img)
            if res:
                OutputSingleLine("The image contains a crack")
            else:
                OutputSingleLine("The image does not contain a crack")

pg.windows.close()