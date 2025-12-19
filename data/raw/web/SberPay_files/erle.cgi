
(function (ph){
try{
var A = self['' || 'AdriverCounter'],
	a = A(ph);
a.reply = {
ph:ph,
rnd:'360039',
bt:62,
sid:223989,
pz:0,
sz:'%2fru%2fperson%2fpayments%2fsberpay%2fnfc',
bn:0,
sliceid:0,
netid:0,
ntype:0,
tns:0,
pass:'',
adid:0,
bid:2864425,
geoid:14,
cgihref:'//ad.adriver.ru/cgi-bin/click.cgi?sid=223989&ad=0&bid=2864425&bt=62&bn=0&pz=0&xpid=DdR4pVRX2e_CHDWH0y71zJcxrjWf823sLsF6P2yIvmppBUruDcOxWxz_R0-fNCt312LaZNJr3RSCNTKcFtAjUdIh4qA&ref=https:%2f%2fwww.sberbank.ru%2fru%2fperson%2fpayments%2fsberpay%2fnfc&custom=',
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
xpid:'DdR4pVRX2e_CHDWH0y71zJcxrjWf823sLsF6P2yIvmppBUruDcOxWxz_R0-fNCt312LaZNJr3RSCNTKcFtAjUdIh4qA'
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
