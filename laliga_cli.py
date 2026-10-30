#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, sys, subprocess

def main():
    p = argparse.ArgumentParser(description='La Liga Intelligence Pipeline - Automated BTTS/Over Predictions')
    p.add_argument('command', choices=['update','predict','golden','next','refresh'], nargs='?', default='golden',
                   help='update=refresh data + predict, predict=run advanced model, golden=print Golden Pick only, next=show fixtures, refresh=refetch CSV')
    p.add_argument('--teamA', help='Team A for manual predict')
    p.add_argument('--teamB', help='Team B for manual predict')
    args = p.parse_args()

    if args.command == 'refresh':
        print('[1/3] Refreshing La Liga data from Understat...')
        subprocess.run([sys.executable, 'C:/Users/User/fetch_laliga.py'], check=False)
        print('[2/3] Running Poisson base...')
        subprocess.run([sys.executable, 'C:/Users/User/btts_poisson_analysis.py'], check=False)
    
    if args.command in ['update','predict','golden','next']:
        # Always ensure latest prediction
        if args.command == 'update':
            subprocess.run([sys.executable, 'C:/Users/User/fetch_laliga.py'], check=False)
        # Run advanced
        if args.teamA and args.teamB:
            import importlib.util
            spec = importlib.util.spec_from_file_location('btts', 'C:/Users/User/btts_poisson_analysis.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            r = mod.predict_btts(args.teamA, args.teamB)
            print(r)
            return
        # Full pipeline
        print('[PIPELINE] Running advanced predictions...')
        subprocess.run([sys.executable, 'C:/Users/User/predict_next_gw_advanced.py'], check=False)
        
        if args.command == 'golden':
            # already prints golden report at end of advanced, also show CSV tail
            import csv
            try:
                rows = list(csv.DictReader(open('laliga_next_gw_btts_predictions_advanced.csv', encoding='utf-8-sig')))
                print('')
                print('CSV Updated: laliga_next_gw_btts_predictions_advanced.csv | %d matches' % len(rows))
            except: pass

if __name__ == '__main__':
    main()
