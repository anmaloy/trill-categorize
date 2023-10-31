
import sys
sys.path.append('..')
from src import proc
import click

@click.command("Example")
@click.argument('name')
@click.option('--n_repeat','-n',help='Number of times to repeat. (Default is 1)',type=int,default=1)
def main(name,n_repeat):
    '''
    This is an example for how to use click command line scripts
    
    Example Usage:

    python example_script_click.py bob

    python example_script_click.py bob -n 10 


    '''
    for ii in range(n_repeat):
        print(name)
    print('Thanks!')

if __name__=='__main__':
    main()