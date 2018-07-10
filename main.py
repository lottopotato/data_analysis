# main

from plot import *
from numpy_process import *
          
def main():
     x, y = load()

     x_arr = list2numpy(x)
     y_arr = list2numpy(y)

     newX_arr = array_division_367(x_arr)

     mean_y = array_division_367(y_arr, "mean")
     var_y = array_division_367(y_arr, "var")
     std_y = array_division_367(y_arr, "std")

     fig, plot = create_fig(2,2)
     scatter(plot[0,0], x_arr, y_arr, "basic", "mediumvioletred", option="add")
     scatter(plot[0,1], newX_arr, mean_y, "mean", "blue", option="add")
     scatter(plot[1,0], newX_arr, var_y, "var", "red", option="add")
     scatter(plot[1,1], newX_arr, std_y, "std", "green", option="add")

     setting_fig(fig)
     plot_show()



if __name__ == "__main__":
     main()
