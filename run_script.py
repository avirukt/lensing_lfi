import sys
import subprocess
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
short_sha = repo.git.rev_parse(sha, short=7)
name = "%s-%s"%(sys.argv[1][:-3],short_sha)

script = "scripts/%s.sh"%name
out = "~/name/outputs/%s.out"%name

with open("template_script.sh", "r") as file:
	f = file.readlines()

def ins(line,s):
	f[line] = f[line][:-1]+s+f[line][-1]

ins(1,name)
ins(2,out)
ins(-1,sys.argv[1] + " " + name)

with open(script, "w") as file:
	for line in f:
		file.write(line)

subprocess.run(["sbatch",script])
