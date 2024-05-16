import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def main():
    # american_rose = (1.0, 0.01, 0.24)
    seadov = (1.0, 0.11, 0.11)
    start = 0.7

    r, g, b = seadov

    mydict = {
        'red': (
            (0.0, 1.0, 1.0),
            (start, r-0.2, 1.0),
            (1.0, r, 0.0),
        ),
        'green': (
            (0.0, 1.0, 1.0),
            (start, 1.0, 1.0),
            (1.0, g, 0.0),
        ),
        'blue': (
            (0.0, 1.0, 1.0),
            (start, 1.0, 1.0),
            (1.0, b, 0.0),
        )
    }
    mpl.colormaps.register(LinearSegmentedColormap('Tulip', mydict))

    def show_colorbar():
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)

        cmap = "Tulip"
        norm = mpl.colors.Normalize(vmin=0, vmax=10)

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax, orientation='horizontal')
        plt.show()
    # show_colorbar()
        
main()