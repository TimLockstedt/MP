from uncertainties import unumpy as up
import uncertainties
import numpy as np

class FFP:
    def __init__(self, data, func, name = "no_name_given", save = True):
        self.__data = data
        self.__func = func
        self.__name = name
        self.set_data()
        self.__save_bool = save
        self.FFP()
        return


    def FFP(self):
        self.FFP_func()
        if self.__save_bool:
            self.save_res()
        try:
            self.FFP_func()
            if self.__save_bool:
                self.save_res()
        except:
            print("Make sure that the given function, only contains the available variables: ",
                (" x"*(len(self.__x) != 0)+
                ", y"*(len(self.__y) != 0)+
                ", z"*(len(self.__z) != 0)+
                ", a"*(len(self.__a) != 0)+
                ", b"*(len(self.__b) != 0)+
                ", c"*(len(self.__c) != 0)
                )
            )
            print("Something went wrong!! \n" +
                    "use FFP.help() for an example Skript and further explination"
                    )
        return


    def help():
        print(
            "#Example Skript:\n"+
            "import numpy as np\n\n"+
            "name = \"Name_of_your_datafile\"\n"+
            "filetype = \".txt\"\n"+
            "#import the given file and saving the data in the variable data\n"+
            "data = np.genfromtxt(name + filetype, skip_header = 0, skip_footer = 0)\n\n"+
            "#Function for calculation of the results, you can use numpy functions etc. just imoprt it beforehand\n"+
            "def func(x = None, y = None, z = None, a = None, b = None, c = None):\n"+
            "\treturn x\n\n"+
            "#define results as an FFP objekt\n"+
            "#save = True (default), means that a txt-file will be created with the res and err in it\n"+
            "results = FFP(data = data, func = func, name = name, save = True)\n"+
            "#List of results:\n"+
            "res = results.get_res()\n"+
            "#List of uncertaintys of the results:\n"+
            "err = results.get_err()\n"
            "#List of result and uncertainty in ufloat format as a list\n"+
            "u_res = results.get_ures()\n"
        )
        return


    def wrapped_f(self):
        return uncertainties.wrap(self.__func)


    def set_data(self):
        self.__x = (up.uarray(self.__data[:, 0], self.__data[:, 1])) if len(self.__data[0]) > 0 else []
        self.__y = (up.uarray(self.__data[:, 2], self.__data[:, 3])) if len(self.__data[0]) > 2 else []
        self.__z = (up.uarray(self.__data[:, 4], self.__data[:, 5])) if len(self.__data[0]) > 4 else []
        self.__a = (up.uarray(self.__data[:, 6], self.__data[:, 7])) if len(self.__data[0]) > 6 else []
        self.__b = (up.uarray(self.__data[:, 8], self.__data[:, 9])) if len(self.__data[0]) > 8 else []
        self.__c = (up.uarray(self.__data[:, 10], self.__data[:, 11])) if len(self.__data[0]) > 10 else []
        return


    def FFP_func(self):
        self.__result = []
        for i in range(len(self.__x)):
            self.__result.append(self.wrapped_f()(
                x = self.__x[i] if len(self.__x) != 0 else 0, 
                y = self.__y[i] if len(self.__y) != 0 else 0, 
                z = self.__z[i] if len(self.__z) != 0 else 0,
                a = self.__a[i] if len(self.__a) != 0 else 0,
                b = self.__b[i] if len(self.__b) != 0 else 0,
                c = self.__c[i] if len(self.__c) != 0 else 0   
                ))
        return
    

    def divide_res(self):
        return np.c_[[self.__result[i].n for i in range(len(self.__result))], [self.__result[i].s for i in range(len(self.__result))]]
    

    def get_data(self):
        return self.__data


    def get_func(self):
        return self.__func


    def get_ures(self):
        return self.__result


    def get_res(self):
        return self.divide_res()[:,0]
        

    def get_err(self):
        return self.divide_res()[:,1]

    def save_res(self):
        liste = self.divide_res()
        if self.__name == "no_name_given":
            print("\nYou can specify the name of the txt-File by using:\nFFP(data,func,name=\"NAME\")")
        np.savetxt(self.__name + '_result.txt', liste , fmt='%r')
        return