# git init from a local directory


```
git init
git remote add origin git@github.com:selous123/libadver.git
#pull origin master->master "assure no duplicate file"
git pull origin master

git add -A
git commit -m "init repo"
git push origin master
 
```


```
#查看当前分支
git branch
#查看远程分支
gitremote
```


# .gitignore untrack file

```
git rm -r --cached .
git rm --cached foo.txt (thanks @amadeann).
```