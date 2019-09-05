# Logging with optional verbosity levels

# Verbosity global constants
SILENT = 0
ERROR = 1
WARN = 2
INFO = 3
DEBUG = 4

# Current verbosity
_curVerbosity = INFO

# Output file
_output = None

# Sets the current output file.
#
# parameters/returns:
#  output : str | None (output filename)
def setCurOutput(output):
    global _output
    _output = output
    f = open(_output, 'w')
    f.close()

# Sets the current verbosity (defaults to standard).
#
# parameters/returns:
#  verbosity : int
def setCurVerbosity(verbosity):
    global _curVerbosity
    _curVerbosity = verbosity

# Prints the given message if its verbosity is
# at most the current setting.
#
# parameters/returns:
#  msg : str
#  verbosity : int
def log(msg, verbosity):
    if verbosity <= _curVerbosity:
        print(msg)
        if not _output is None:
            f = open(_output, 'a')
            f.write(msg + '\n')
            f.close()
