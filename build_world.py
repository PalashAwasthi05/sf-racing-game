"""General reconstruction entry point; all location data is supplied at runtime."""
import argparse
from general_world.run import execute

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='Normalized source-measurement manifest');p.add_argument('--output',required=True,help='Candidate and diagnostic output directory');a=p.parse_args();execute(a.input,a.output)
