
import os, sys
sys.path.append("..")

PROFILER_STATS_PATH = os.path.join(os.getcwd(), 'profiler')

_generate_cProfile_cmd = "python3 -m cProfile -o " + \
    os.path.join(PROFILER_STATS_PATH, 'profile.pstats') + \
    " -s module main.py"

os.system(_generate_cProfile_cmd)

_generate_pGraph_cmd = "gprof2dot -f pstats " + \
    os.path.join(PROFILER_STATS_PATH, 'profile.pstats') + \
    " > " + \
    os.path.join(PROFILER_STATS_PATH, 'map.dot')


os.system(_generate_pGraph_cmd)

_show_ptable_cmd = "cprofilev -f " + \
    os.path.join(PROFILER_STATS_PATH, 'profile.pstats') + \
    " | start chrome http://127.0.0.1:4000"

os.system(_show_ptable_cmd)

