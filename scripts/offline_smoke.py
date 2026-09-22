"""Run an actual replay with Python socket connections disabled (not an OS firewall)."""
import os,socket,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'

def blocked(*args,**kwargs):raise RuntimeError('Network use is disabled in this offline smoke run')

socket.socket.connect=blocked;socket.socket.connect_ex=blocked
socket.create_connection=blocked;socket.getaddrinfo=blocked
from prts_core.demo import main
if __name__=='__main__':main()
