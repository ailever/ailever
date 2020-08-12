# built-in modules / external modules
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json

# ailever modules
import options

json.load(open('prediction_{options.id}.json'))
