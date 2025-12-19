
(function (ph){
try{
var A = self['' || 'AdriverCounter'],
	a = A(ph);
a.reply = {
ph:ph,
rnd:'649700',
bt:62,
sid:223989,
pz:0,
sz:'%2fru%2fperson%2fpayments%2fsberpay%2fonline_sberpay',
bn:0,
sliceid:0,
netid:0,
ntype:0,
tns:0,
pass:'',
adid:0,
bid:2864425,
geoid:14,
cgihref:'//ad.adriver.ru/cgi-bin/click.cgi?sid=223989&ad=0&bid=2864425&bt=62&bn=0&pz=0&xpid=D8QnmskP9Vf-JXttA1XmG8GUciUcJ6j1lBRaTxSGEw3l7BrrU4H_b3RVISM1W7-84SU3R835j6VMwIcH1NveWG1MaIA&ref=https:%2f%2fwww.sberbank.ru%2fru%2fperson%2fpayments%2fsberpay%2fonline_sberpay&custom=',
target:'_blank',
width:'0',
height:'0',
alt:'AdRiver',
mirror:A.httplize('//mlb3.adriver.ru'), 
comp0:'0/script.js',
custom:{},
track_site:0,
cid:'ANOHU0TM8kPOL-Guq-R0ONw',
uid:2336612072883,
xpid:'D8QnmskP9Vf-JXttA1XmG8GUciUcJ6j1lBRaTxSGEw3l7BrrU4H_b3RVISM1W7-84SU3R835j6VMwIcH1NveWG1MaIA'
}
var r = a.reply;

r.comppath = r.mirror + '/images/0002864/0002864425/' + (/^0\//.test(r.comp0) ? '0/' : '');
r.comp0 = r.comp0.replace(/^0\//,'');
if (r.comp0 == "script.js" && r.adid){
	A.defaultMirror = r.mirror; 
	A.loadScript(r.comppath + r.comp0 + '?v' + ph) 
} else if ("function" === typeof (A.loadComplete)) {
   A.loadComplete(a.reply);
}
}catch(e){} 
}('1'));
