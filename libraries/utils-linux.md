## TMUX

`~/.tmux.conf`
```bash
# changing prefix from 'Ctrl+b' to 'Ctrl+a'

unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix
```
