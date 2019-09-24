%economic load dispatch using simulated annealing technique
clear;
ftdatan;
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
end
pd=750;
dpm=1;
ni1=0;
while (dpm~=0)
    ni1=ni1+1;
    for n=1:(nt-1)
        pt(n)=unifrnd(ptmn(n),ptmx(n));
    end
    pgt=0;
    for n=1:(nt-1)
        pgt=pgt+pt(n);
    end
    pt(nt)=pd-pgt;   
    for n=1:2
        if ((pt(nt)>=ptmn(nt))&(pt(nt)<=ptmx(nt)))
            dpm=0;            
        else 
            dpm=1;
            pt(nt)=0;
        end
    end   
end
ni1,dpm
tfc=0;
for n=1:nt
    c=fc(n)*(ptmn(n)-pt(n));
    d=ec(n)*sin(c);
    tfc=tfc+ac(n)+bc(n)*pt(n)+cc(n)*pt(n)^2+abs(d);
end 
tm=tfc;
tfc
ni2=0;
while (tm>=10)
    ni2=ni2+1;
    for ni3=1:30
        dpm=1;
        while (dpm~=0)
            for n=1:(nt-1)
                sgm=tm*0.001;
                change=normrnd(0,sgm)
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
                if ((pt1(nt)>=ptmn(nt))&(pt1(nt)<=ptmx(nt)))
                    dpm=0;                    
                else
                    dpm=1;
                    pt1(nt)=0;
                end
            end           
        end
        tfc1=0;
        for n=1:nt
            c=fc(n)*(ptmn(n)-pt1(n));
            d=ec(n)*sin(c);
            tfc1=tfc1+ac(n)+bc(n)*pt1(n)+cc(n)*pt1(n)^2+abs(d);
        end       
        dtfc=tfc1-tfc;
%         rand=unifrnd(0,1);
%         dt=-(dtfc/tm);
%         rc=exp(dt);
        if (dtfc<=0)
            for n=1:nt
                pt(n)=pt1(n);
            end
            tfc=tfc1;            
         else
            for n=1:nt
                pt(n)=pt(n);
            end
            tfc=tfc;
        end
     end
     %tm=0.98*tm;
     tm=0.95*tm  
     sol(ni2)=tfc;
     tfc
 end 
 tfc=0;
 for n=1:nt
     c=fc(n)*(ptmn(n)-pt(n));
     d=ec(n)*sin(c);
     tfc=tfc+ac(n)+bc(n)*pt(n)+cc(n)*pt(n)^2+abs(d);
 end     
pt
tfc
X=1:ni2;
Y=sol(X);
plot(X,Y,'k-')
xlabel('iteration')
ylabel('cost($)')
