#! /usr/bin/env python3
'''
FAVITES: FrAmework for VIral Transmission and Evolution Simulation
'''
import argparse
import os
from glob import glob
from os import chdir,getcwd,makedirs,remove
from os.path import abspath,expanduser,isdir,isfile
from shutil import copyfile,move,rmtree
from subprocess import call,check_output,CalledProcessError,DEVNULL,STDOUT
from sys import platform,stderr,stdout
from warnings import warn
from urllib.error import URLError
from urllib.request import urlopen
DOCKER_IMAGE = "docker://niemasd/favites"
MAIN_VERSION_SYMBOLS = {'0','1','2','3','4','5','6','7','8','9','.'}
INCOMPATIBLE = {'1.0.0','1.0.1','1.0.2','1.0.3','1.1.0','1.1.1','1.1.2','1.1.3','1.1.4','1.1.5','1.1.6'}
FAVITES_DIR = expanduser('~/.favites')

# return True if the given tag (string) is a main version (e.g. '1.1.1') or False if not (e.g. '1.1.1a')
def is_main_version(tag):
    for c in tag:
        if c not in MAIN_VERSION_SYMBOLS:
            return False
    return True

# get the latest FAVITES Docker image main version
def get_latest_version():
    try:
        DOCKER_TAGS = list(); curr_url = "https://hub.docker.com/v2/repositories/niemasd/favites/tags/?page=1"
        while curr_url is not None:
            tmp = eval(urlopen(curr_url).read().decode('utf-8').replace(': null',': None').replace(': true',': True'))
            DOCKER_TAGS += [e['name'] for e in tmp['results']]
            curr_url = tmp['next']
        DOCKER_TAGS = [tag for tag in DOCKER_TAGS if is_main_version(tag)] # remove non-main-version
        DOCKER_TAGS = [tuple(int(i) for i in tag.split('.')) for tag in DOCKER_TAGS] # convert to tuple of ints
        DOCKER_TAGS.sort() # sort in ascending order
        return '.'.join(str(i) for i in DOCKER_TAGS[-1])
    except Exception as e:
        raise RuntimeError("Failed to use Python 3 urllib to connect to FAVITES Docker repository webpage\n%s"%str(e))

# parse user args
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config', required=True, type=str, help="Configuration file")
parser.add_argument('-o', '--out_dir', required=False, type=str, help="Output directory")
parser.add_argument('-s', '--random_number_seed', required=False, type=int, help="Random number seed")
parser.add_argument('-v', '--verbose', action="store_true", help="Print verbose messages to stderr")
parser.add_argument('-u', '--update', required=True, nargs='*', help="Update Docker image (-u to pull newest version, -u <VERSION> to pull <VERSION>)")
args = parser.parse_args()

# check core user args
DELETE_AFTER = list()
CONFIG = abspath(expanduser(args.config))
assert isfile(CONFIG), "ERROR: Cannot open configuration file: %s" % CONFIG
try:
    CONFIG_DICT = eval(open(CONFIG).read())
except:
    raise SyntaxError("Malformed FAVITES configuration file. Must be valid JSON")
if args.out_dir is not None:
    if 'out_dir' in CONFIG_DICT:
        warn("Output directory specified in command line (%s) and config file (%s). Command line will take precedence" % (args.out_dir, CONFIG_DICT['out_dir']))
    CONFIG_DICT['out_dir'] = args.out_dir
assert 'out_dir' in CONFIG_DICT, "Parameter 'out_dir' is not in the configuration file!"
OUTPUT_DIR = abspath(expanduser(CONFIG_DICT['out_dir']))

# create output directory
try:
    makedirs(OUTPUT_DIR)
except:
    if isdir(OUTPUT_DIR):
        response = 'x'
        while len(response) == 0 or response[0] not in {'y','n'}:
            response = input("ERROR: Output directory exists. Overwrite? All contents will be deleted. (y/n) ").strip().lower()
            if response[0] == 'y':
                rmtree(OUTPUT_DIR); makedirs(OUTPUT_DIR)
            else:
                exit(-1)

# check other user args
if args.random_number_seed is not None:
    if "random_number_seed" in CONFIG_DICT:
        warn("Random number seed specified in command line (%d) and config file (%s). Command line will take precedence" % (args.random_number_seed, CONFIG_DICT['random_number_seed']))
    CONFIG_DICT["random_number_seed"] = args.random_number_seed
if "random_number_seed" not in CONFIG_DICT:
    CONFIG_DICT["random_number_seed"] = ""
CN_FILE = None
if 'contact_network_file' in CONFIG_DICT:
    CN_FILE = abspath(expanduser(CONFIG_DICT['contact_network_file']))
    CONFIG_DICT['contact_network_file'] = '/FAVITES_MOUNT/%s' % CN_FILE.split('/')[-1]
    DELETE_AFTER.append('%s/USER_CN.TXT' % OUTPUT_DIR)
    copyfile(CN_FILE, DELETE_AFTER[-1])
TN_FILE = None
if 'transmission_network_file' in CONFIG_DICT:
    TN_FILE = abspath(expanduser(CONFIG_DICT['transmission_network_file']))
    CONFIG_DICT['transmission_network_file'] = '/FAVITES_MOUNT/%s' % TN_FILE.split('/')[-1]
    DELETE_AFTER.append('%s/USER_TN.TXT' % OUTPUT_DIR)
    copyfile(TN_FILE, DELETE_AFTER[-1])
TREE_FILE = None
if 'tree_file' in CONFIG_DICT:
    TREE_FILE = abspath(expanduser(CONFIG_DICT['tree_file']))
    CONFIG_DICT['tree_file'] = '/FAVITES_MOUNT/%s' % TREE_FILE.split('/')[-1]
    DELETE_AFTER.append('%s/USER_TREE.TRE' % OUTPUT_DIR)
    copyfile(TREE_FILE, DELETE_AFTER[-1])
ERRORFREE_SEQ_FILE = None
if 'errorfree_sequence_file' in CONFIG_DICT:
    ERRORFREE_SEQ_FILE = abspath(expanduser(CONFIG_DICT['errorfree_sequence_file']))
    CONFIG_DICT['errorfree_sequence_file'] = '/FAVITES_MOUNT/%s' % ERRORFREE_SEQ_FILE.split('/')[-1]
    DELETE_AFTER.append('%s/USER_SEQS.FAS' % OUTPUT_DIR)
    copyfile(ERRORFREE_SEQ_FILE, DELETE_AFTER[-1])
DELETE_AFTER.append('%s/USER_CONFIG.JSON' % OUTPUT_DIR)
TMP_CONFIG = open(DELETE_AFTER[-1],'w')
TMP_CONFIG.write(str(CONFIG_DICT).replace(": inf",": float('inf')"))
TMP_CONFIG.close()

# pull the newest versioned Docker image (if applicable)
if args.update is not None:
    assert len(args.update) < 2, "More than one Docker image version specified. Must either specify just -u or -u <VERSION>"
    if len(args.update) == 0:
        tag = get_latest_version()
    else:
        tag = args.update[0]
        assert tag not in INCOMPATIBLE, "Using incompatible version (%s). Singularity is only supported in FAVITES 1.1.7 onward"%tag
    version = '%s:%s'%(DOCKER_IMAGE,tag)

# first pull Docker image as Singularity image
pulled_image = '%s/singularity-favites-%s.img' % (FAVITES_DIR,tag)
if not isfile(pulled_image):
    makedirs(FAVITES_DIR, exist_ok=True)
    orig_dir = getcwd()
    chdir(FAVITES_DIR)
    print("Pulling Docker image (%s)..." % tag, end=' '); stdout.flush()
    try:
        COMMAND = ['singularity','pull','--name',pulled_image,version]
        check_output(COMMAND, stderr=DEVNULL)
    except:
        raise RuntimeError("singularity pull command failed: %s" % ' '.join(COMMAND))
    chdir(orig_dir)
    print("done"); stdout.flush()

# set up Docker command and run
COMMAND =  ['singularity','run','-e']              # Singularity command
COMMAND += ['-B',OUTPUT_DIR+':/FAVITES_MOUNT:rw']  # mount output directory
COMMAND += [pulled_image]                          # Docker image
try:
    call(COMMAND)
except:
    raise RuntimeError("singularity run command failed: %s" % ' '.join(COMMAND))

# clean up temporary files
for f in DELETE_AFTER:
    remove(f)
for f in glob('%s/OUTPUT_DIR/*'%OUTPUT_DIR):
    move(f, '%s/%s'%(OUTPUT_DIR, f.split('/')[-1]))
rmtree('%s/OUTPUT_DIR'%OUTPUT_DIR)
