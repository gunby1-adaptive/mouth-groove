s=open('src/template.html').read()
s=s.replace('/*ENGINE*/',open('src/engine.js').read()).replace('/*SOUNDS*/',open('src/sounds.js').read())
open('index.html','w').write(s)
print(len(s))
