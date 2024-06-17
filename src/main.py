import os
import math
import random
from tkinter import *
from tkinter import ttk
from pyautogui import size as sz
import cv2
import numpy as np
from numpy import add as add
from PIL import Image, ImageTk
from data import logo


class Particle:
    def __init__(self):
        self.N = 1000
        self.P = []
        self.W = [1/self.N] * self.N
        self.prev = [1/self.N] * self.N

    def init_particles(self, w, h):
        for i in range(self.N):
            self.P.append(np.array([random.uniform(-(w-1)/2, (w-1)/2), random.uniform(-(h-1)/2, (h-1)/2)]))

    def move_particles(self, u, std, xy2px):
        for (i, p) in list(enumerate(self.P)):
            u_noise = np.array([u[0] + random.gauss(0, std), u[1] + random.gauss(0, std)])
            p[0] += xy2px(u_noise)[0]
            p[1] += xy2px(u_noise)[1]

    def weigh(self, w, h, mymap, agent_hist, crop_img):
        temp = np.zeros(self.N)
        out = [' '] * self.N

        for (i, p) in list(enumerate(self.P)):
            if self.W[i] == 0:
                continue
            if (p[0] < -(w-1)*0.6) | (p[0] > (w-1)*0.6) | (p[1] < -(h-1)*0.6) | (p[1] > (h-1)*0.6):
                out[i] = 'x'
                continue
            ref = crop_img(mymap, p)
            ref_hist = cv2.calcHist([ref], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            temp[i] = 1 - cv2.compareHist(agent_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)

        total = np.sum(temp)
        if total == 0:
            self.W = [1 / self.N] * self.N
        else:
            for (j, w) in list(enumerate(self.W)):
                if out[j] == 'x':
                    self.W[j] = 0
                    continue
                if w == 0:
                    continue
                self.W[j] = np.interp(temp[j], [0, total], [0, 1])*0.5 + self.W[j]*0.3 + self.prev[j]*0.2
                if self.W[j] < 0.8/self.N:
                    self.W[j] = 0

        new_total = np.sum(self.W)
        for k in range(len(self.W)):
            self.W[k] = self.W[k] / new_total

        self.prev = self.W.copy()


class Localization(Particle):
    def __init__(self, particle):
        Particle.__init__(self)
        self.particle = particle

        # Initial GUI setup
        W, H = sz()
        self.master = Tk()
        self.master.title("Particle Filter Localization")
        self.master.tk.call("wm", "iconphoto", self.master._w, PhotoImage(data=logo))
        self.master.geometry("{}x{}+{}+{}".format(round(W*0.8), round(H*0.8), round(W*0.1), round(H*0.1)))
        self.master.resizable(False, False)
        self.master.bind("<Key>", self.keyPressed)
        self.cv = Canvas(self.master)
        self.cv.pack(fill=BOTH, expand=YES)
        self.fr = ttk.Frame(self.cv)
        self.fr.pack(fill=BOTH, expand=YES)
        self.c1 = Canvas(self.cv, bd=-2)
        self.c2 = Canvas(self.cv, bd=-2)

        # Create images
        self.map1 = map1 = cv2.cvtColor(cv2.imread(self.resource_path('map1.png')), cv2.COLOR_BGR2RGB)
        self.map2 = map2 = cv2.cvtColor(cv2.imread(self.resource_path('map2.png')), cv2.COLOR_BGR2RGB)
        self.map3 = map3 = cv2.cvtColor(cv2.imread(self.resource_path('map3.png')), cv2.COLOR_BGR2RGB)
        self.master.wait_visibility(self.fr)
        m1 = cv2.resize(map1, (int(map1.shape[1]*self.fr.winfo_width()*0.2/map1.shape[1]), int(map1.shape[0]*self.fr.winfo_width()*0.2/map1.shape[1])))
        m1 = ImageTk.PhotoImage(image=Image.fromarray(m1))
        m2 = cv2.resize(map2, (int(map2.shape[1]*self.fr.winfo_width()*0.2/map2.shape[1]), int(map2.shape[0]*self.fr.winfo_width()*0.2/map2.shape[1])))
        m2 = ImageTk.PhotoImage(image=Image.fromarray(m2))
        m3 = cv2.resize(map3, (int(map3.shape[1]*self.fr.winfo_height()*0.2/map3.shape[0]), int(map3.shape[0]*self.fr.winfo_height()*0.2/map3.shape[0])))
        m3 = ImageTk.PhotoImage(image=Image.fromarray(m3))

        # Create buttons
        self.var = IntVar()
        ttk.Button(self.fr, image=m1, command=self.b1).place(relx=0.1, rely=0.4, relwidth=0.2, relheight=0.2)
        ttk.Button(self.fr, image=m2, command=self.b2).place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.2)
        ttk.Button(self.fr, image=m3, command=self.b3).place(relx=0.7, rely=0.4, relwidth=0.2, relheight=0.2)
        self.master.wait_variable(self.var)
        self.c2.create_line(0, 0, 0, H * 0.8, width=3)

        # Declare variables
        self.mymap = self.fit_img(self.mymap)
        self.master.img = img = ImageTk.PhotoImage(image=Image.fromarray(self.mymap))
        self.h, self.w = self.mymap.shape[:2]
        self.m = 51
        self.s = 50
        self.d = 1
        self.std = 0.1
        self.ofs1 = np.array([self.c1.winfo_width()/2, self.c1.winfo_height()/2])
        self.ofs2 = np.array([self.c2.winfo_width()/2, self.c2.winfo_height()/2])
        self.pos_xy = np.array([random.uniform(-(self.w-self.m+1)/self.s/2, (self.w-self.m+1)/self.s/2), random.uniform(-(self.h-self.m+1)/self.s/2, (self.h-self.m+1)/self.s/2)])
        self.pos_px = self.xy2px(self.pos_xy)
        self.bel_xy = np.array([self.pos_xy[0], self.pos_xy[1]])
        self.bel_px = self.xy2px(self.bel_xy)
        self.u = np.array([0, 0])

        # Display agent
        self.c1.create_rectangle(self.ofs1[0] - self.w/2, self.ofs1[1] - self.h/2, self.ofs1[0] + self.w/2, self.ofs1[1] + self.h/2, width=10)
        self.c1.create_image(self.ofs1[0], self.ofs1[1], image=img, anchor=CENTER)
        self.agent1 = self.c1.create_oval(add(self.ofs1, self.pos_px)[0] - 10, add(self.ofs1, self.pos_px)[1] - 10, add(self.ofs1, self.pos_px)[0] + 10, add(self.ofs1, self.pos_px)[1] + 10, fill="#00FF00")
        self.agent2 = self.c1.create_rectangle(add(self.ofs1, self.pos_px)[0] - (self.m-1)/2, add(self.ofs1, self.pos_px)[1] - (self.m-1)/2, add(self.ofs1, self.pos_px)[0] + (self.m-1)/2, add(self.ofs1, self.pos_px)[1] + (self.m-1)/2, width=5)
        self.line1 = self.c1.create_line(add(self.ofs1, self.pos_px)[0], self.ofs1[1] - (self.h-1)/2, add(self.ofs1, self.pos_px)[0], add(self.ofs1, self.pos_px)[1] - (self.m-1)/2, width=5)
        self.line2 = self.c1.create_line(add(self.ofs1, self.pos_px)[0], add(self.ofs1, self.pos_px)[1] + (self.m-1)/2, add(self.ofs1, self.pos_px)[0], self.ofs1[1] + (self.h-1)/2, width=5)
        self.line3 = self.c1.create_line(self.ofs1[0] - (self.w-1)/2, add(self.ofs1, self.pos_px)[1], add(self.ofs1, self.pos_px)[0] - (self.m-1)/2, add(self.ofs1, self.pos_px)[1], width=5)
        self.line4 = self.c1.create_line(add(self.ofs1, self.pos_px)[0] + (self.m - 1) / 2, add(self.ofs1, self.pos_px)[1], self.ofs1[0] + (self.w - 1) / 2, add(self.ofs1, self.pos_px)[1], width=5)

        # Crop image and display
        self.r = self.c2.winfo_width() * 0.5 / self.m
        self.cropped = self.crop_img(self.mymap, self.pos_px)
        self.rs_cropped = cv2.resize(self.cropped, (int(self.cropped.shape[1]*self.r), int(self.cropped.shape[0]*self.r)), interpolation=cv2.INTER_AREA)
        self.crop = crop = ImageTk.PhotoImage(image=Image.fromarray(self.rs_cropped))
        self.disp_crop = self.c2.create_image(self.ofs2[0], self.ofs2[1], image=crop, anchor=CENTER)
        self.agent_hist = cv2.calcHist([self.cropped], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])

        # Display particles
        particle.init_particles(self.w, self.h)
        for (i, p) in list(enumerate(particle.P)):
            if particle.W[i] < 5 / particle.N:
                r = np.interp(particle.W[i], [0, 5/particle.N], [0, 20])
            else:
                r = np.interp(particle.W[i], [0.01, 1], [50, 200])
            self.c1.create_oval(add(self.ofs1, p)[0] - r, add(self.ofs1, p)[1] - r, add(self.ofs1, p)[0] + r, add(self.ofs1, p)[1] + r, tags="p" + hex(i), fill="#FF0000")
        self.c1.lift(self.agent1)
        self.c1.lift(self.agent2)
        self.c1.lift(self.line1)
        self.c1.lift(self.line2)
        self.c1.lift(self.line3)
        self.c1.lift(self.line4)

    def b1(self):
        self.fr.destroy()
        self.c1.place(relx=0, rely=0, relwidth=0.75, relheight=1)
        self.c2.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)
        self.mymap = self.map1
        self.var.set(1)

    def b2(self):
        self.fr.destroy()
        self.c1.place(relx=0, rely=0, relwidth=0.75, relheight=1)
        self.c2.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)
        self.mymap = self.map2
        self.var.set(1)

    def b3(self):
        self.fr.destroy()
        self.c1.place(relx=0, rely=0, relwidth=0.75, relheight=1)
        self.c2.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)
        self.mymap = self.map3
        self.var.set(1)

    def xy2px(self, xy):
        px = np.array([int(round(xy[0] * self.s)), int(round(xy[1] * self.s))])
        return px

    def fit_img(self, image):
        self.master.wait_visibility(self.c1)
        s = 0.75
        img_h, img_w = image.shape[:2]
        if image.shape[1] > image.shape[0]:
            r = self.c1.winfo_width() * s / float(img_w)
            dim = (2 * math.floor(int(self.c1.winfo_width() * s) / 2) + 1, 2 * math.floor(int(img_h * r) / 2) + 1)
        else:
            r = self.c1.winfo_height() * s / float(img_h)
            dim = (2 * math.floor(int(img_w * r) / 2) + 1, 2 * math.floor(int(self.c1.winfo_height() * s) / 2) + 1)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def crop_img(self, image, pos):
        real_pos = np.array([pos[0]+(self.w-1)/2, pos[1]+(self.h-1)/2])
        c = image[int(real_pos[1]-(self.m-1)/2):int(real_pos[1]+(self.m-1)/2+1), int(real_pos[0]-(self.m-1)/2):int(real_pos[0]+(self.m-1)/2+1)]
        if c.shape[0] < self.m:
            c = np.concatenate((c, np.zeros((101-c.shape[0], c.shape[1], 3), np.uint8)), axis=0)
        if c.shape[1] < self.m:
            c = np.concatenate((c, np.zeros((c.shape[0], 101-c.shape[1], 3), np.uint8)), axis=1)
        return c

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def keyPressed(self, event):
        if (event.char != " ") & (event.keysym != "Return"):
            return

        while 1:
            dx = random.choice([-1, 1]) * random.uniform(0, self.d)
            dy = random.choice([-1, 1]) * math.sqrt(self.d - dx**2)
            u_true = np.array([dx + random.gauss(0, self.std), dy + random.gauss(0, self.std)])
            if (self.pos_xy[0] + u_true[0] >= -(self.w-self.m+1)/self.s/2) & (self.pos_xy[0] + u_true[0] < (self.w-self.m+1)/self.s/2) & (self.pos_xy[1] + u_true[1] >= -(self.h-self.m+1)/self.s/2) & (self.pos_xy[1] + u_true[1] < (self.h-self.m+1)/self.s/2):
                self.u = np.array([dx, dy])
                self.bel_xy[0] += self.u[0]
                self.bel_xy[1] += self.u[1]
                self.pos_xy[0] += u_true[0]
                self.pos_xy[1] += u_true[1]
                self.bel_px = self.xy2px(self.bel_xy)
                self.pos_px = self.xy2px(self.pos_xy)
                break

        # Update display on left canvas
        self.c1.coords(self.agent1, add(self.ofs1, self.pos_px)[0] - 10, add(self.ofs1, self.pos_px)[1] - 10, add(self.ofs1, self.pos_px)[0] + 10, add(self.ofs1, self.pos_px)[1] + 10)
        self.c1.coords(self.agent2, add(self.ofs1, self.pos_px)[0] - (self.m-1)/2, add(self.ofs1, self.pos_px)[1] - (self.m-1)/2, add(self.ofs1, self.pos_px)[0] + (self.m-1)/2, add(self.ofs1, self.pos_px)[1] + (self.m-1)/2)
        self.c1.coords(self.line1, add(self.ofs1, self.pos_px)[0], self.ofs1[1] - (self.h-1)/2, add(self.ofs1, self.pos_px)[0], add(self.ofs1, self.pos_px)[1] - (self.m-1)/2)
        self.c1.coords(self.line2, add(self.ofs1, self.pos_px)[0], add(self.ofs1, self.pos_px)[1] + (self.m-1)/2, add(self.ofs1, self.pos_px)[0], self.ofs1[1] + (self.h-1)/2)
        self.c1.coords(self.line3, self.ofs1[0] - (self.w-1)/2, add(self.ofs1, self.pos_px)[1], add(self.ofs1, self.pos_px)[0] - (self.m-1)/2, add(self.ofs1, self.pos_px)[1])
        self.c1.coords(self.line4, add(self.ofs1, self.pos_px)[0] + (self.m - 1) / 2, add(self.ofs1, self.pos_px)[1], self.ofs1[0] + (self.w - 1) / 2, add(self.ofs1, self.pos_px)[1])

        # Update display on right canvas
        self.c2.delete(self.disp_crop)
        self.cropped = self.crop_img(self.mymap, self.pos_px)
        self.rs_cropped = cv2.resize(self.cropped, (int(self.cropped.shape[1]*self.r), int(self.cropped.shape[0]*self.r)), interpolation=cv2.INTER_AREA)
        self.crop = crop = ImageTk.PhotoImage(image=Image.fromarray(self.rs_cropped))
        self.disp_crop = self.c2.create_image(self.ofs2[0], self.ofs2[1], image=crop, anchor=CENTER)

        # Move/Resize particles
        self.agent_hist = cv2.calcHist([self.cropped], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        self.particle.move_particles(self.u, self.std, self.xy2px)
        self.particle.weigh(self.w, self.h, self.mymap, self.agent_hist, self.crop_img)
        for (i, p) in list(enumerate(self.particle.P)):
            if self.particle.W[i] < 5 / self.particle.N:
                r = np.interp(self.particle.W[i], [0, 5/self.particle.N], [0, 20])
            else:
                r = np.interp(self.particle.W[i], [0.01, 1], [50, 200])
            self.c1.coords("p" + hex(i), add(self.ofs1, p)[0] - r, add(self.ofs1, p)[1] - r, add(self.ofs1, p)[0] + r, add(self.ofs1, p)[1] + r)


if __name__ == "__main__":
    particles = Particle()
    app = Localization(particles)
    app.master.mainloop()