import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import Image,ImageTk
import os

import pollen_detector as pd

class Gui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.geometry("900x500")
        self.title("Pollen Detector Hackathon by group Cool Beans")
        self.mainFrame = tk.Frame(self)
        self.inputFrame = tk.Frame(self.mainFrame)
        self.outputFrame = tk.Frame(self.mainFrame)
        self.mainFrame.grid(column=0,row=0,sticky="news")
        self.inputFrame.grid(column=0,row=0)
        self.outputFrame.grid(column=0,row=1)

        # input frame
        self.inputFileButton = tk.Button(self.inputFrame,text="File",command=self.getImage,width=20,padx=0)
        self.inputFileNameLabel = tk.Label(self.inputFrame,text="filename:")
        self.inputFileName = tk.Entry(self.inputFrame,width=50,state="disabled")
        self.runButton = tk.Button(self.inputFrame,text="Run",width=20,padx=0,command=self.run)

        self.inputFileButton.grid(column=0,row=0)
        self.inputFileNameLabel.grid(column=1,row=0)
        self.inputFileName.grid(column=2,row=0)
        self.runButton.grid(column=3,row=0)

        self.inputFilePath = None
        
        
        # output frame
        self.lightPollenLabel = tk.Label(self.outputFrame,text="Light pollen")
        self.darkPollenLabel = tk.Label(self.outputFrame,text="Dark pollen")
        self.totalPollenLabel = tk.Label(self.outputFrame,text="Total pollen")
        self.lightPollen = tk.Entry(self.outputFrame,width=50,state="disabled")
        self.darkPollen = tk.Entry(self.outputFrame,width=50,state="disabled")
        self.totalPollen = tk.Entry(self.outputFrame,width=50,state="disabled")

        self.lightPollenLabel.grid(column=0,row=0)
        self.darkPollenLabel.grid(column=1,row=0)
        self.totalPollenLabel.grid(column=2,row=0)
        self.lightPollen.grid(column=0,row=1)
        self.darkPollen.grid(column=1,row=1)
        self.totalPollen.grid(column=2,row=1)

        # 
        self.outputImageWidget = tk.Label(self.outputFrame)
        self.outputImageWidget.grid(column=0,row=2,columnspan=3)
        self.outputImageLabel = tk.Label(self.outputFrame)
        self.outputImageLabel.grid(column=0,row=3,columnspan=3)
    
    def getImage(self): 
        # file dialog
        self.inputFilePath = fd.askopenfilename(
            filetypes=[("JPG Files","*.jpg"),("All Files","*.*")],
            initialdir=os.getcwd()
        )
        if not self.inputFilePath: return

        self.inputFileName.config(state="normal")
        self.inputFileName.delete(0,tk.END)
        self.inputFileName.insert(0,os.path.basename(self.inputFilePath))
        self.inputFileName.config(state="disabled")

    def run(self):
        if not self.inputFilePath:
            messagebox.showwarning("Warning","No input file selected!")
            return
        self.outputFilePath = "./output/_allpollen.jpg"
        self.detector = pd.PollenDetector()
        self.detector.params["hough"]["minRadius"] = 20
        hcircles, contours = self.detector.detect(self.inputFilePath)

        # Change the text in Entry
        self.lightPollen.config(state="normal")
        self.darkPollen.config(state="normal")
        self.totalPollen.config(state="normal")
        
        self.lightPollen.delete(0,tk.END)
        self.darkPollen.delete(0,tk.END)
        self.totalPollen.delete(0,tk.END)

        self.lightPollen.insert(0,self.detector.lightPollen)
        self.darkPollen.insert(0,self.detector.darkPollen)
        self.totalPollen.insert(0,self.detector.pollenCount)

        self.lightPollen.config(state="disabled")
        self.darkPollen.config(state="disabled")
        self.totalPollen.config(state="disabled")

        self.detector.drawContours(self.inputFilePath, contours)
        self.detector.drawHoughCircles(self.inputFilePath, hcircles)
        
        # display the image
        self.geometry("1000x600")
        self.outputImage = ImageTk.PhotoImage(Image.open(self.outputFilePath).resize((1000,500),Image.ANTIALIAS))
        self.outputImageWidget.config(image=self.outputImage)
        self.outputImageLabel.config(text=os.path.basename(self.outputFilePath))

window = Gui()
window.mainloop()