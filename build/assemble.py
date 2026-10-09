import json
t=open('template.html').read(); d=open('data.json').read()
assert '/*__DATA__*/null' in t
open('../index.html','w').write(t.replace('/*__DATA__*/null', d))
print('index.html', len(t)+len(d))
