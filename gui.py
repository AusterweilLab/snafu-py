import Tkinter as Tk

# the constructor syntax is:
# OptionMenu(master, variable, *values)

OPTIONS = [
    "egg",
    "bunny",
    "chicken"
]

root = Tk.Tk()

variable = Tk.StringVar(root)
variable.set(OPTIONS[0]) # default value

w = apply(Tk.OptionMenu, (root, variable) + tuple(OPTIONS))
w.pack()

root.mainloop()
