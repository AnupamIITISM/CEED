clear;
tic;
 tdata=[150 470 786.7988 38.5397 0.1524 450 0.041 103.3908 -2.4444 0.0312 0.5035 0.0207
        135 470 451.3251 46.1591 0.1058 600 0.036 103.3908 -2.4444 0.0312 0.5035 0.0207
        73 340 1049.9977 40.3965 0.0280 320 0.028 300.3910 -4.0695 0.0509 0.4968 0.0202 
        60 300 1243.5311 38.3055 0.0354 260 0.052 300.3910 -4.0695 0.0509 0.4968 0.0202
        73 243 1658.5696 36.3278 0.0211 280 0.063 320.0006 -3.8132 0.0344 0.4972 0.0200
        57 160 1356.6592 38.2704 0.0179 310 0.048 320.0006 -3.8132 0.0344 0.4972 0.0200
        020 130 1450.7045 36.5104 0.0121 300 0.086 330.0056 -3.9023 0.0465 0.5163 0.0214
        47 120 1450.7045 36.5104 0.0121 340 0.082 330.0056 -3.9023 0.0465 0.5163 0.0214
        20 80 1455.6056 39.5804 0.1090 270 0.098 350.0056 -3.9524 0.0465 0.5475 0.0234
        10 55 1469.4026 40.5407 0.1295 380 0.094 360.0012 -3.9864 0.0470 0.5475 0.0234];
    
B=[0.000049 0.000014 0.000015 0.000015 0.000016 0.000017 0.000017 0.000018 0.000019 0.000020
0.000014 0.000045 0.000016 0.000016 0.000017 0.000015 0.000015 0.000016 0.000018 0.000018
0.000015 0.000016 0.000039 0.000010 0.000012 0.000012 0.000014 0.000014 0.000016 0.000016
0.000015 0.000016 0.000010 0.000040 0.000014 0.000010 0.000011 0.000012 0.000014 0.000015
0.000016 0.000017 0.000012 0.000014 0.000035 0.000011 0.000013 0.000013 0.000015 0.000016
0.000017 0.000015 0.000012 0.000010 0.000011 0.000036 0.000012 0.000012 0.000014 0.000015
0.000017 0.000015 0.000014 0.000011 0.000013 0.000012 0.000038 0.000016 0.000016 0.000018
0.000018 0.000016 0.000014 0.000012 0.000013 0.000012 0.000016 0.000040 0.000015 0.000016
0.000019 0.000018 0.000016 0.000014 0.000015 0.000014 0.000016 0.000015 0.000042 0.000019
0.000020 0.000018 0.000016 0.000015 0.000016 0.000015 0.000018 0.000016 0.000019 0.000044];

%=====================================================================
%economic load dispatch using simulated annealing technique
% clear;
% ftdatan;

nt=size(tdata,1);
for m=1:nt
   ptmn(m)=tdata(m,1);
   ptmx(m)=tdata(m,2);
   ac(m)=tdata(m,3);
   bc(m)=tdata(m,4);
   cc(m)=tdata(m,5);
   ec(m)=tdata(m,6);
   fc(m)=tdata(m,7);
   ae(m)=tdata(m,8);
   be(m)=tdata(m,9);
   ce(m)=tdata(m,10);
   ee(m)=tdata(m,11);
   fe(m)=tdata(m,12);  
   
   B11=B(m,1);
   B1n=B(m,2);
   Bnn=B(m,3:nt);   
end
pd=1036;
% dpm=1;
ni1=0;

% while (dpm~=0)
    ni1=ni1+1;
    for n=1:(nt-1)
        pt(n)=unifrnd(ptmn(n),ptmx(n));
    end
    pgt=0;
    for n=1:(nt-1)
        pgt=pgt+pt(n);
    end
    pt(nt)=pd-pgt;  
    
A=B11;
BB1=2*B1n*pgt';
B1=BB1-1;
C1=pgt*Bnn*pgt';
C=pd-pgt+C1;
ptx1=roots([A B1 C]);
ptx=abs(min(ptx1));
%  pt(nt) = pd-pgt+C1;
Pl=pgt*B*pgt';
lam=abs(sum(pgt)-pd-pgt*B*pgt');
 
    for n=1:2
        if ((pt(nt)>=ptmn(nt))&(pt(nt)<=ptmx(nt)))
%             dpm=0;            
        else 
%             dpm=1;
            pt(nt)=0;
        end
    end   
% end
ni1
% dpm
tfc=0;
tfe=0;
E=0;
% w2=0;
% w2=0;
% input('w2 = ')
% w1=1-w2;
for n=1:nt
    hi=(ac(n)+ bc(n)*ptmx(n)+ cc(n) * (ptmx(n)^2))/(ce(n)+ be(n)*ptmx(n)+ ae(n) * (ptmx(n)^2));
    c = fc(n)*(ptmn(n)-pt(n));
    d=ec(n)*sin(c);
    tfc=tfc+ac(n)+bc(n)*pt(n)+cc(n)*pt(n)^2+abs(d);
    dc=fe(n)*pt(n);
    tfe=tfe+ae(n)+be(n)*pt(n)+ce(n)*pt(n)^2+ee(n)*exp(dc);
    E= hi * tfc + (1-hi)*tfe;
end 
tm=E;
E
ni2=0;
while (tm>=1e-4)
    ni2=ni2+1;
    for ni3=1:30
%         dpm=1;
%         while (dpm~=0)
            for n=1:(nt-1)
                sgm=tm*0.01;
                change=normrnd(0,sgm);
                pt1(n)=pt(n)+normrnd(0,sgm);
                if (pt1(n)>ptmx(n))
                    pt1(n)=ptmx(n);
                elseif (pt1(n)<ptmn(n))
                    pt1(n)=ptmn(n);
                end
            end
            pgt=0;           
            for n=1:(nt-1)
                pgt=pgt+pt1(n);
            end           
            pt1(nt)=pd-pgt;           
            for n=1:2
                if ((pt1(nt)<=ptmn(nt))&(pt1(nt)>=ptmx(nt)))
%                 if (~((pt1(nt)>=ptmn(nt))&(pt1(nt)<=ptmx(nt))))
%                     dpm=0;                    
%                 else
%                     dpm=1;
                    pt1(nt)=0;
                end
            end           
%         end
        tfc1=0;
        tfe1=0;
        E1=0;
%         w21=0;
%         w11=1-w21;
%         input('w21 = ')
        for n=1:nt
            A=B11;
BB1=2*B1n*pgt';
B1=BB1-1;
C1=pgt*Bnn*pgt';
C=pd-pgt+C1;
ptx=roots([A B1 C]);
 pt(nt)=abs(min(ptx));
 
%  pt(nt) = pd-pgt+C1;
Pl=pgt*B*pgt';
lam=abs(sum(pgt)-pd-pgt*B*pgt');

            hi1=(ac(n)+ bc(n)*ptmx(n)+ cc(n) * (ptmx(n)^2))/(ce(n)+ be(n)*ptmx(n)+ ae(n) * (ptmx(n)^2));
            c=fc(n)*(ptmn(n)-pt1(n));
            d=ec(n)*sin(c);
            tfc1=tfc1+ac(n)+bc(n)*pt1(n)+cc(n)*pt1(n)^2+abs(d);
            dc=fe(n)*pt1(n);
            tfe1=tfe1+ae(n)+be(n)*pt1(n)+ce(n)*pt1(n)^2+ee(n)*exp(dc);
            E1= hi1 * tfc1 + (1-hi1)* tfe1;
        end       
        dtfc=tfc1 - tfc;
        dtfe=tfe1 - tfe;
        dtfEtotal = E1-E;
        rand=unifrnd(0,1);
%         dt=-(dtfc/tm);
%         dte=-(dtfe/tm);
        dt = -(dtfEtotal/tm);
        rc=exp(dt);
        if (dtfc<=0 && dtfe<=0 && dtfEtotal<=0)
            for n=1:nt
                pt(n)=pt1(n);
            end
            tfc=tfc1;
            tfe=tfe1;
            E=E1;
         else
            for n=1:nt
                pt(n)=pt(n);
            end
            tfc=tfc;
            tfe=tfe;
            E=E;
        end
     end
     tm=0.95*tm;
     sol(ni2)=E;     
     sol2(ni2)=tfc;     
     sol3(ni2)=tfe;     
 end
 tfc=0;
 tfe=0;
 E=0;
 for n=1:nt
     A=B11;
BB1=2*B1n*pgt';
B1=BB1-1;
C1=pgt*Bnn*pgt';
C=pd-pgt+C1;
ptx=roots([A B1 C]);
 pt(nt)=abs(min(ptx));
 
%  pt(nt) = pd-pgt+C1;
Pl=pgt*B*pgt';
lam=abs(sum(pgt)-pd-pgt*B*pgt');

     hi=[ptmx(n)^2*cc(n)+bc(n)*ptmx(n)+ac(n)]/[ce(n)+be(n)*ptmx(n)+ae(n)*ptmx(n)^2];
     c=fc(n)*(ptmn(n)-pt(n));
     d=ec(n)*sin(c);
     tfc=tfc+ac(n)+bc(n)*pt(n)+cc(n)*pt(n)^2+abs(d);
     dc=fe(n)*pt(n);
     tfe=tfe+ae(n)+be(n)*pt(n)+ce(n)*pt(n)^2+ee(n)*exp(dc);
     E= hi * tfc + (1-hi)*tfe;
 end 
 tfc = tfc+1000*lam;
 tfc
 tfe
pt
E

X=1:ni2;
Y=sol(X);
plot(X,Y,'k-')
xlabel('iteration')
ylabel('E')
figure

Y1=sol2(X);
plot(X,Y1)
xlabel('iteration')
ylabel('tfc')
figure

Y2=sol3(X);
plot(X,Y2)
xlabel('iteration')
ylabel('tfe')
toc;