import os

workers = int(os.environ.get('WEB_CONCURRENCY', 1))
threads = int(os.environ.get('WEB_THREADS', 2))
timeout = 120
bind = "0.0.0.0:10000"