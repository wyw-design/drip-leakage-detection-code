max=0;
t=0;

pause(3);                               
global vid;                           
vid = videoinput('winvideo',2,'YUY2_640x480');        %Start the CCD camera and set the resolution
set(vid,'ReturnedColorSpace','rgb');                
figure(1);
image0=image(getsnapshot(vid));         
preview(vid,image0);              

pause(2);                             
imagePrevious0=getsnapshot(vid);      
I{1}=imagePrevious0;
dingshi=300;                            %Set a time for video shooting
for k=2:dingshi
    pause(0.00033);                        
    imageCurrent0=getsnapshot(vid);                      
    I{k} = imageCurrent0;   
    imagePrevious=I{k-1};
    imageCurrent=I{k};
    Imback= double(imagePrevious);
    Imwork= double(imageCurrent);  
    cc(k)= 0;
    [MR,MC,Dim] = size(Imback);
    %Subtract the input image from the background image to get the area with the greatest difference
    fore = zeros(MR,MC);
    fore = (abs(Imwork(:,:,1)-Imback(:,:,1)) > 10) ...
     | (abs(Imwork(:,:,2) - Imback(:,:,2)) > 10) ...
     | (abs(Imwork(:,:,3) - Imback(:,:,3)) > 10);  
    foremm = bwmorph(fore,'erode',2); 
    labeled = bwlabel(foremm,4);      
    stats = regionprops(labeled,['basic']);
    [N,W] = size(stats);
    if N < 1
       continue   
    end
    %If the number of large spots is greater than 1, the bubbling method is used to sort them (from largest to smallest )
    id = zeros(N);     
    for i = 1 : N
        id(i) = i;
    end
    for i = 1 : N-1
        for j = i+1 : N
            if stats(i).Area < stats(j).Area
               tmp = stats(i);
               stats(i) = stats(j);
               stats(j) = tmp;
               tmp = id(i);
               id(i) = id(j);
               id(j) = tmp;
            end
        end
     end
     if stats(1).Area < 100 
         continue
     end
     selected = (labeled==id(1));
     % Obtain the center and radius of the maximum spot area, and set the flag to 1.
     centroid = stats(1).Centroid;
     cc(k)= centroid(1);             %Centroid row coordinates
     if k==2
         Im1 = imageCurrent;
         max = cc(k);
      end      
      if (cc(k)>max)
         Im1 = imageCurrent;
         max = cc(k);
         t=k;
         if k>4
            Im2=I{t-3};
         end
      end      
end
closepreview;                    %Turn off the camera 
% figure(2);
% subplot(2,1,1);
% imshow(Im1);              
% title('background image');
% subplot(2,1,2);
% imshow(Im2);              
% title('critical image');
               
set(0,'defaultfigurecolor','w');
RGB1=Im2;
figure(1),subplot(121),imshow(RGB1),title('critical image');
I1=rgb2gray(RGB1);
X=I1;                
figure(2),subplot(121),imshow(X),title(' Grayscale critical image with noise ');    
X=double(X);
[c,l]=wavedec2(X,2,'coif2');         
n=[1,2];                                             
p=[10.28,24.08];                     % Set the threshold vector,
nc=wthcoef2('h',c,l,n,p,'s');       
X1=waverec2(nc,l,'coif2');           
mc=wthcoef2('v',nc,l,n,p,'s');     
X2=waverec2(mc,l,'coif2');          
X2=uint8(X2);
figure(3),subplot(121),imshow(X2),title(' Critical image after secondary denoising ');

I=X2;
row=size(I,1);
column=size(I,2);
N=zeros(1, 256);
for i=1:row
    for j=1:column
        k=I(i, j);
        N(k+1)=N(k+1)+1;
    end
end

figure(4),subplot(121),bar(N),axis tight,title('The histogram of the critical image ');

I=double(I);
J=(I-60)*255/50;   
row=size(I,1);
column=size(I,2);
for i=1:row
    for j=1:column
        if J(i, j)<0
            J(i, j)=0;
        end
        if J(i, j)>255;
            J(i, j)=255;
        end
    end
end
X3=uint8(J);
figure(5),subplot(121),imshow(X3),title(' Critical image after image enhancement ');

BW1=im2bw(X3,0.1);
% level=graythresh(X2);
% BW1=im2bw(X2,level);
figure(6),subplot(121),imshow(BW1),title('Critical binary image');


BW1=1-BW1;
BW1=bwareaopen(BW1,2000);

figure(7),subplot(121),imshow(BW1),title('Critical binary image after Border Reflectance Elimination');
se1=strel('disk',6);    
BW1=imdilate(BW1,se1);    
figure(8),subplot(121),imshow(BW1),title('Critical binary Imag of expansion');
se2=strel('disk',6);
BW1=imerode(BW1,se2);     
figure(9),subplot(121),imshow(BW1),title('Binary imag of corrosion threshold');

BW1=1-BW1;
BW1=bwareaopen(BW1,15000);
AA0=BW1;
figure(10),subplot(121),imshow(BW1),title('Solidified critical binary image');

%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
RGB2=Im1;
figure(1),subplot(122),imshow(RGB2),title('background image');
I2=rgb2gray(RGB2);
XX=I2;                
figure(2),subplot(122),imshow(XX),title(' Grayscale background image with noise ');    
XX=double(XX);
[c,l]=wavedec2(XX,2,'coif2');       
n=[1,2];                                           
p=[10.28,24.08];                      
nc=wthcoef2('h',c,l,n,p,'s');        
XX1=waverec2(nc,l,'coif2');                   
mc=wthcoef2('v',nc,l,n,p,'s');       
XX2=waverec2(mc,l,'coif2');         
XX2=uint8(XX2);
figure(3),subplot(122),imshow(XX2),title(' Background Image after Secondary Denoising ');

II=XX2;
rowW=size(II,1);
columnN=size(II,2);
NN=zeros(1, 256);
for i=1:rowW
    for j=1:columnN
        kK=II(i, j);
        NN(kK+1)=NN(kK+1)+1;
    end
end

figure(4),subplot(122),bar(NN),axis tight,title('The histogram of background image ');

II=double(II);
JJ=(II-60)*255/50;   
rowW=size(II,1);
columnN=size(II,2);
for i=1:rowW
    for j=1:columnN
        if JJ(i, j)<0
            JJ(i, j)=0;
        end
        if JJ(i, j)>255;
            JJ(i, j)=255;
        end
    end
end
XX3=uint8(JJ);
figure(5),subplot(122),imshow(XX3),title(' Background Image after Image Enhancement ');

BW2=im2bw(XX3,0.1);
figure(6),subplot(122),imshow(BW2),title('Background Binary Image');

BW2=1-BW2;
BW2=bwareaopen(BW2,2000);
figure(7),subplot(122),imshow(BW2),title('Background Binarization Map after Boundary Reflectance Elimination');

se1=strel('disk',6);    
BW2=imdilate(BW2,se1);    
figure(8),subplot(122),imshow(BW2),title('Dilated background binary image');
se2=strel('disk',6);
BW2=imerode(BW2,se2);     
figure(9),subplot(122),imshow(BW2),title('Corrosion background binary image');

BW2=1-BW2;
BW2=bwareaopen(BW2,15000);
A0=BW2;
figure(10),subplot(122),imshow(BW2),title('Solid background binary image');
%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
BW1=1-BW1;
BW2=1-BW2;
BW=BW1-BW2;
BW=bwareaopen(BW,1500);
BW=1-BW;
A1=BW;
figure(11),imshow(BW),title('Approximation of Droplet Binary Images Using Difference Method');
%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%------Collect concave curve information-------%
[m,n]=size(A0);
flag=0;  
exist=0; 
count=0; 
p=1;     
for i=1:m
    for j=1:n
        if A0(i,j)==0&count==0
           B0(p,1,1)=i;    
           B0(p,2,1)=j;    
           count=count+1; 
           flag=1;        
           exist=1;
        end
        if A0(i,j)==1&count==1
           B0(p,1,2)=i;     
           B0(p,2,2)=j-1;   
           count=count+1;  
           break;          
        end
    end
    if flag==1             
       p=p+1;
       count=0;
       flag=0; 
    elseif flag==0&exist==1
       break;              
    end
end 
p0=p-1;%p0=82;B0(82,1,1)=82
% B0(82,,1)=(82,325)
% B0(82,,2)=(82,339)
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%------Prepare for the determination of the central axis-------%
[m,n]=size(AA0);
flag=0;  
exist=0; 
count=0; 
p=1;     %
for i=1:m
    for j=1:n
        if AA0(i,j)==0&count==0
           B2(p,1,1)=i;    
           B2(p,2,1)=j;    
           count=count+1;  
           flag=1;         
           exist=1;
        end
        if AA0(i,j)==1&count==1
           B2(p,1,2)=i;     
           B2(p,2,2)=j-1;   
           count=count+1;  
           break;           
        end
    end
    if flag==1             
       p=p+1;
       count=0;
       flag=0; 
    elseif flag==0&exist==1
       break;              
    end
end 
pzz=p-1;  %pzz=214

% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%-----Determine the length of the transverse semi-axis-----%    
C0=0;
for i=1:p0   
   R(i,1)=(B0(i,2,2)-B0(i,2,1))./2; 
   C0=C0+(B0(i,2,1)+R(i,1));      
end
C0=round(C0./p0);   

C1=0;
for i=1:(pzz)   
   R(i,1)=(B2(i,2,2)-B2(i,2,1))./2; 
   C1=C1+(B2(i,2,1)+R(i,1));     
end
C1=round(C1./(pzz));     %C1=335, and the central axis is 335
C=round((C0+C1)/2);
% C=334
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%------Determine the subject information to be collected-------%
[m,n]=size(A1);    %A1=BW
flag=0;  
exist=0; 
p=1;    
for i=1:m
    for j=1:n
        if A1(i,j)==0
           B1(p,1,1)=i;    
           B1(p,2,1)=j;    
           flag=1;         
           exist=1;
        end
    end
    if flag==1             
       p=p+1;
       flag=0; 
    elseif flag==0&exist==1
       break;             
    end
end 
p1=p-1;  %p1=154
% B1(1,,1)=(61,300)
% B1(154,,1)=(214,334)
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%-----Extract contour points-----%
% Segment (line):61-82(22)(1-22);83-214(132)(23-154)
[m,n]=size(A1);                    %%%%m=480,n=640
flag=0;  
exist=0; 
count=0; 
t=1;     
for i=1:B0(p0,1,1)                   
    for j=1:n
        if A1(i,j)==0&count==0
            B(t,1,1)=i;      %B(1,,1)=(61,300)
            B(t,2,1)=j;      %B(22,,1)=(82,297)
            count=count+1;   
            flag=1;          
            exist=1;
        end
        if A1(i,j)==1&count==1
            B(t,1,2)=i;       %B(1,,2)=(61,300)
            B(t,2,2)=j-1;   %B(22,,2)=(82,324)
            count=count+1;  
            break;          
        end
    end
    if flag==1 
        t=t+1;
        count=0;
        flag=0; 
    elseif flag==0&exist==1
        break; 
    end
end 

flag=0;  
exist=0; 
count=0; 
for i=(B0(p0,1,1)+1):m                     
    for j=1:n
        if A1(i,j)==0&count==0
            B(t,1,1)=i;          %B(23,,1)=(83,297)
            B(t,2,1)=j;          %B(154,,1)=(214,329)
            count=count+1; 
            flag=1;        
            exist=1;
        end
        if A1(i,j)==1&count==1
            B(t,1,2)=i;            %B(23,,2)=(83,373)
            B(t,2,2)=j-1;        %B(154,,2)=(214,334)
            count=count+1;  
            break;   
        end
    end
    if flag==1  
        t=t+1;
        count=0;
        flag=0; 
    elseif flag==0&exist==1
        break;  
    end
end 
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%-----Determining the first point that appears on the left-----%  
que=B0(p0,1,1)-B1(1,1,1)+1;%que=82-61+1=22
kk=1;
for i=1:que
    if B(i,2,1)>C
        i=i+1;
    else
        kk=i;  %kk=1,and left points total 22
        break  %return
    end
end
ppp=p1;
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
for i=kk:que   %1-22
    R(i)=C-B(i,2,2);                         
end
%-----Obtaining the Coefficients of Each Fitted Function Expression-----%      Concave left half formula
clear y;
x=B1(1,1,1):B0(p0,1,1);    %61-82
cha=que-kk+1;
for i=1:(cha+kk-1)                %1:22
    y(i)=C-R(i,1);    
end
pp1=polyfit(x,y,4);        %The best fit is a quadric fit
ypp1=polyval(pp1,x);

% figure(12);
% p11=polyfit(x,y,1);
% y11=polyval(p11,x);
% subplot(3,2,1);
% plot(y,x,'r',y11,x,'b');   
% axis ij
% % title('first fit');
% p12=polyfit(x,y,2);
% y12=polyval(p12,x);
% subplot(3,2,2);
% plot(y,x,'r',y12,x,'b');  
% axis ij
% % title('quadratic fit');
% p13=polyfit(x,y,3);
% y13=polyval(p13,x);
% subplot(3,2,3);
% plot(y,x,'r',y13,x,'b');  
% axis ij
% % title('cubic fit');
% p14=polyfit(x,y,4);
% y14=polyval(p14,x);
% subplot(3,2,4);
% plot(y,x,'r',y14,x,'b');  
% axis ij
% % title('quadric fit');
% p15=polyfit(x,y,5);
% y15=polyval(p15,x);
% subplot(3,2,5);
% plot(y,x,'r',y15,x,'b');  
% axis ij
% % title('quintic fit');
% p16=polyfit(x,y,6);
% y16=polyval(p16,x);
% subplot(3,2,6);
% plot(y,x,'r',y16,x,'b');  
% axis ij
% % title('Sixth-order fit');

% for i=1:cha
%     u11(i)=y(i)-y11(i);
%     u12(i)=y(i)-y12(i);
%     u13(i)=y(i)-y13(i);
%     u14(i)=y(i)-y14(i);
%     u15(i)=y(i)-y15(i);
%     u16(i)=y(i)-y16(i);
% end
% 
% %E(X)
% u11x=sum(u11)./cha;
% u12x=sum(u12)./cha;
% u13x=sum(u13)./cha;
% u14x=sum(u14)./cha;
% u15x=sum(u15)./cha;
% u16x=sum(u16)./cha;
% %E(X2)
% u11x2=sum(u11.^2)./cha;
% u12x2=sum(u12.^2)./cha;
% u13x2=sum(u13.^2)./cha;
% u14x2=sum(u14.^2)./cha;
% u15x2=sum(u15.^2)./cha;
% u16x2=sum(u16.^2)./cha;
% %b--D=E(X2)-E(X)2
% u11b=sqrt(u11x2-u11x.^2);
% u12b=sqrt(u12x2-u12x.^2);
% u13b=sqrt(u13x2-u13x.^2);
% u14b=sqrt(u14x2-u14x.^2);
% u15b=sqrt(u15x2-u15x.^2);
% u16b=sqrt(u16x2-u16x.^2);
% 
% x=B1(1,1,1):B0(p0,1,1); 
% x1=B1(1,1,1):0.01:B0(p0,1,1); 

% figure(13),subplot(3,2,1),plot(x1,u11b,'b.');  
% hold on
% figure(13),subplot(3,2,1),plot(x1,u11x,'g.');  
% hold on
% figure(13),subplot(3,2,1),plot(x,u11,'ro ');%,title('Primary residual map');
% 
% figure(13),subplot(3,2,2);,plot(x1,u12b,'b.');  
% hold on
% figure(13),subplot(3,2,2);,plot(x1,u12x,'g.');  
% hold on
% figure(13),subplot(3,2,2),plot(x,u12,'ro ');%,title('Quadratic residual map');
% 
% figure(13),subplot(3,2,3);,plot(x1,u13b,'b.');  
% hold on
% figure(13),subplot(3,2,3);,plot(x1,u13x,'g.');  
% hold on
% figure(13),subplot(3,2,3),plot(x,u13,'ro ');%,title('Cubic residual map');
% 
% figure(13),subplot(3,2,4);,plot(x1,u14b,'b.');  
% hold on
% figure(13),subplot(3,2,4);,plot(x1,u14x,'g.');  
% hold on
% figure(13),subplot(3,2,4),plot(x,u14,'ro ');%,title('Quartic residual map');
% 
% figure(13),subplot(3,2,5);,plot(x1,u15b,'b.');  
% hold on
% figure(13),subplot(3,2,5);,plot(x1,u15x,'g.');  
% hold on
% figure(13),subplot(3,2,5),plot(x,u15,'ro ');%,title('Quintic residual map');
% 
% figure(13),subplot(3,2,6);,plot(x1,u16b,'b.');  
% hold on
% figure(13),subplot(3,2,6);,plot(x1,u16x,'g.');  
% hold on
% figure(13),subplot(3,2,6),plot(x,u16,'ro ');%,title('Sextic Residual map');
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%%%%%---Analysis of the main body's left half contour, determining inflection points---%%%%%% 
p=p1;
for i=1:p1   %154 numbers
    R(i)=C-B(i,2,1);    
    R1(i)=R(i);      
end

%-----Calculate the linear regression coefficients of each point and its four surrounding points.-----% 
c=ones(4,1);
for i=5:(p1-4)        
p1=[i-4;i-3;i-2;i-1];
p2=[i+1;i+2;i+3;i+4];
y1=[R1(i-4);R1(i-3);R1(i-2);R1(i-1)];
y2=[R1(i+1);R1(i+2);R1(i+3);R1(i+4)];
x1=[c,p1]; 
x2=[c,p2]; 
[b,bint,r,rint,stats]=regress(y1,x1);
q(i-4,1)=b(2); %Obtaining the Regression Coefficients of 4 Points Before Point i
[b,bint,r,rint,stats]=regress(y2,x2);
q(i-4,2)=b(2); %Obtaining the Regression Coefficients of 4 Points After Point i
end 
%-----Computation of first order difference and second order difference-----%   
for i=1:(p1-8)         
    ss1(i,1)=q(i,2)-q(i,1); %First order difference
end
for i=1:(p1-9)        
    ss2(i,1)=ss1(i+1,1)-ss1(i,1); %Second order difference
end
%-----Drawing First and Second Order Difference Graph Lines-----%  Finding Inflection Points Based on Differential Curve
% figure(14);  
% i=5:(p1-4);  
% % subplot(1,2,1);
% plot(i,ss1);
% % title('First-order Difference Diagram of Boundary Curve Slope');
% figure(15); 
% j=5:(p1-5); 
% % subplot(1,2,2);
% plot(j,ss2);
% % title('Second-Order Difference Diagram of Boundary Curve Slope');

%-----Find the turning point-----% 
clear p1
p1=p;
gd=0;
biz=0;
for i=1:p1
    if (B(i,2,2)-B(i,2,1))>biz
       gd=i;
       biz=B(i,2,2)-B(i,2,1);
    end
end
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%-----The main part obtains the coefficients of each fitted function expression.-----%  
% % % % % % Left Piecewise Expression 1
clear y;
x=B(1,1,1):B(gd,1,1);    %61-158
cha=que-kk+1;
for i=1:gd                %1:98
    y(i)=C-R(i+kk-1,1);    %Change back to reference as coordinate system
end
pp2=polyfit(x,y,3);        %Optimal Fitting, Here Chosen as Cubic Fitting
ypp2=polyval(pp2,x);

% figure(16);
% p21=polyfit(x,y,1);
% y21=polyval(p21,x);
% subplot(3,2,1);
% plot(y,x,'r',y21,x,'b');   
% axis ij
% % title('First fit');
% p22=polyfit(x,y,2);
% y22=polyval(p22,x);
% subplot(3,2,2);
% plot(y,x,'r',y22,x,'b');  
% axis ij
% % title('quadratic fit');
% p23=polyfit(x,y,3);
% y23=polyval(p23,x);
% subplot(3,2,3);
% plot(y,x,'r',y23,x,'b');  
% axis ij
% % title('Cubic fit');
% p24=polyfit(x,y,4);
% y24=polyval(p24,x);
% subplot(3,2,4);
% plot(y,x,'r',y24,x,'b');  
% axis ij
% % title('Quartic fit');
% p25=polyfit(x,y,5);
% y25=polyval(p25,x);
% subplot(3,2,5);
% plot(y,x,'r',y25,x,'b');  
% axis ij
% % title('Quintic fit');
% p26=polyfit(x,y,6);
% y26=polyval(p26,x);
% subplot(3,2,6);
% plot(y,x,'r',y26,x,'b');  
% axis ij
% % title('Sixth-order fit');

% for i=1:gd
%     u21(i)=y(i)-y21(i);
%     u22(i)=y(i)-y22(i);
%     u23(i)=y(i)-y23(i);
%     u24(i)=y(i)-y24(i);
%     u25(i)=y(i)-y25(i);
%     u26(i)=y(i)-y26(i);
% end
% 
% %E(X)
% u21x=sum(u21)./cha;
% u22x=sum(u22)./cha;
% u23x=sum(u23)./cha;
% u24x=sum(u24)./cha;
% u25x=sum(u25)./cha;
% u26x=sum(u26)./cha;
% %E(X2)
% u21x2=sum(u21.^2)./cha;
% u22x2=sum(u22.^2)./cha;
% u23x2=sum(u23.^2)./cha;
% u24x2=sum(u24.^2)./cha;
% u25x2=sum(u25.^2)./cha;
% u26x2=sum(u26.^2)./cha;
% %b--D=E(X2)-E(X)2
% u21b=sqrt(u21x2-u21x.^2);
% u22b=sqrt(u22x2-u22x.^2);
% u23b=sqrt(u23x2-u23x.^2);
% u24b=sqrt(u24x2-u24x.^2);
% u25b=sqrt(u25x2-u25x.^2);
% u26b=sqrt(u26x2-u26x.^2);
% 
% x=B(1,1,1):B(gd,1,1); 
% x1=B(1,1,1):0.01:B1(gd,1,1); 

% figure(17),subplot(3,2,1),plot(x1,u21b,'b.');  
% hold on
% figure(17),subplot(3,2,1),plot(x1,u21x,'g.');  
% hold on
% figure(17),subplot(3,2,1),plot(x,u21,'ro ');%,title('Primary residual map');
% 
% figure(17),subplot(3,2,2);,plot(x1,u22b,'b.');  
% hold on
% figure(17),subplot(3,2,2);,plot(x1,u22x,'g.');  
% hold on
% figure(17),subplot(3,2,2),plot(x,u22,'ro ');%,title('Quadratic residual map');
% 
% figure(17),subplot(3,2,3);,plot(x1,u23b,'b.');  
% hold on
% figure(17),subplot(3,2,3);,plot(x1,u23x,'g.');  
% hold on
% figure(17),subplot(3,2,3),plot(x,u23,'ro ');%,title('Cubic residual map');
% 
% figure(17),subplot(3,2,4);,plot(x1,u24b,'b.');  
% hold on
% figure(17),subplot(3,2,4);,plot(x1,u24x,'g.');  
% hold on
% figure(17),subplot(3,2,4),plot(x,u24,'ro ');%,title('Quartic residual map');
% 
% figure(17),subplot(3,2,5);,plot(x1,u25b,'b.');  
% hold on
% figure(17),subplot(3,2,5);,plot(x1,u25x,'g.');  
% hold on
% figure(17),subplot(3,2,5),plot(x,u25,'ro ');%,title('Quintic residual map');
% 
% figure(17),subplot(3,2,6);,plot(x1,u26b,'b.');  
% hold on
% figure(17),subplot(3,2,6);,plot(x1,u26x,'g.');  
% hold on
% figure(17),subplot(3,2,6),plot(x,u26,'ro ');%,title('Sextic residual map');
% % % % % % Left Piecewise Expression 2
clear y;
x=B(gd+1,1,1):B(p1,1,1);    %159-214
cha=que-kk+1;
for i=1:(p1-gd)                %1:56
    y(i)=C-R(i+gd-1,1);    %Change back to reference as coordinate system
end
pp3=polyfit(x,y,4);        %Optimal Fit, Here Chosen as Quadratic Fit
ypp3=polyval(pp3,x);

% figure(18);
% p31=polyfit(x,y,1);
% y31=polyval(p31,x);
% subplot(3,2,1);
% plot(y,x,'r',y31,x,'b');   
% axis ij
% % title('First fit');
% p32=polyfit(x,y,2);
% y32=polyval(p32,x);
% subplot(3,2,2);
% plot(y,x,'r',y32,x,'b');  
% axis ij
% % title('Quadratic fit');
% p33=polyfit(x,y,3);
% y33=polyval(p33,x);
% subplot(3,2,3);
% plot(y,x,'r',y33,x,'b');  
% axis ij
% % title('Cubic fit');
% p34=polyfit(x,y,4);
% y34=polyval(p34,x);
% subplot(3,2,4);
% plot(y,x,'r',y34,x,'b');  
% axis ij
% % title('Quartic fit');
% p35=polyfit(x,y,5);
% y35=polyval(p35,x);
% subplot(3,2,5);
% plot(y,x,'r',y35,x,'b');  
% axis ij
% % title('Quintic fit');
% p36=polyfit(x,y,6);
% y36=polyval(p36,x);
% subplot(3,2,6);
% plot(y,x,'r',y36,x,'b');  
% axis ij
% % title('Sixth-order fit');

% for i=1:(p1-gd)
%     u31(i)=y(i)-y31(i);
%     u32(i)=y(i)-y32(i);
%     u33(i)=y(i)-y33(i);
%     u34(i)=y(i)-y34(i);
%     u35(i)=y(i)-y35(i);
%     u36(i)=y(i)-y36(i);
% end
% 
% %E(X)
% u31x=sum(u31)./cha;
% u32x=sum(u32)./cha;
% u33x=sum(u33)./cha;
% u34x=sum(u34)./cha;
% u35x=sum(u35)./cha;
% u36x=sum(u36)./cha;
% %E(X2)
% u31x2=sum(u31.^2)./cha;
% u32x2=sum(u32.^2)./cha;
% u33x2=sum(u33.^2)./cha;
% u34x2=sum(u34.^2)./cha;
% u35x2=sum(u35.^2)./cha;
% u36x2=sum(u36.^2)./cha;
% %b--D=E(X2)-E(X)2
% u31b=sqrt(u31x2-u31x.^2);
% u32b=sqrt(u32x2-u32x.^2);
% u33b=sqrt(u33x2-u33x.^2);
% u34b=sqrt(u34x2-u34x.^2);
% u35b=sqrt(u35x2-u35x.^2);
% u36b=sqrt(u36x2-u36x.^2);
% 
% x=B(gd+1,1,1):B(p1,1,1); 
% x1=B(gd+1,1,1):0.01:B1(p1,1,1); 

% figure(19),subplot(3,2,1),plot(x1,u31b,'b.');  
% hold on
% figure(19),subplot(3,2,1),plot(x1,u31x,'g.');  
% hold on
% figure(19),subplot(3,2,1),plot(x,u31,'ro ');%,title('Primary residual map');
% 
% figure(19),subplot(3,2,2);,plot(x1,u32b,'b.');  
% hold on
% figure(19),subplot(3,2,2);,plot(x1,u32x,'g.');  
% hold on
% figure(19),subplot(3,2,2),plot(x,u32,'ro ');%,title('Quadratic residual map');
% 
% figure(19),subplot(3,2,3);,plot(x1,u33b,'b.');  
% hold on
% figure(19),subplot(3,2,3);,plot(x1,u33x,'g.');  
% hold on
% figure(19),subplot(3,2,3),plot(x,u33,'ro ');%,title('Cubic residual map');
% 
% figure(19),subplot(3,2,4);,plot(x1,u34b,'b.');  
% hold on
% figure(19),subplot(3,2,4);,plot(x1,u34x,'g.');  
% hold on
% figure(19),subplot(3,2,4),plot(x,u34,'ro ');%,title('Quartic residual map');
% 
% figure(19),subplot(3,2,5);,plot(x1,u35b,'b.');  
% hold on
% figure(19),subplot(3,2,5);,plot(x1,u35x,'g.');  
% hold on
% figure(19),subplot(3,2,5),plot(x,u35,'ro ');%,title('Quintic residual map');
% 
% figure(19),subplot(3,2,6);,plot(x1,u36b,'b.');  
% hold on
% figure(19),subplot(3,2,6);,plot(x1,u36x,'g.');  
% hold on
% figure(19),subplot(3,2,6),plot(x,u36,'ro ');%,title('Sextic residual map');

% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%--Mapping of isolated droplets with approximate droplet boundary maps obtained from our side-fitted curves--%
% Segment (line):61-82 (1-22) (22)
%The actual processing row number on the left side is 153.
xt=zeros(m,n);%(480,640);     
r1=round(C-ypp1);     
r2=round(C-ypp2);
r3=round(C-ypp3);
% % % % % %Segmented Expression Fit Curve Diagram
for i=B(1,1,1):B(cha,1,1)  %61-82
   xt(i,C-r1(i-B(1,1,1)+1))=1;
   xt(i,C+r1(i-B(1,1,1)+1))=1; %Draw the boundary of the top 
end 
for i=B(1,1,1):B(gd,1,1)  %61-158
xt(i,C-r2(i-B(1,1,1)+1))=1;
xt(i,C+r2(i-B(1,1,1)+1))=1; %Draw the boundary of the first section.
end
for i=B(gd+1,1,1):B(p1,1,1) %159=214
xt(i,C-r3(i-B(gd,1,1)))=1;
xt(i,C+r3(i-B(gd,1,1)))=1; %Draw the boundary of the second section.
end 

% % % % % %Trace the first line, seal
for i=C-r2(1):C+r2(1) 
    xt(B(1,1,1),i)=1; 
end 
for i=C-r1(1):C+r1(1) 
    xt(B(1,1,1),i)=0; 
end 
% % % % % %Trace the bottom of the concave curve and seal it.
for i=C-r1(cha):C+r1(cha) 
    xt(B(cha,1,1),i)=1; 
end 
% % % % % % %Trace the last line and seal it.
for i=C-r3(p1-gd):C+r3(p1-gd) 
    xt(B(p1,1,1),i)=1; 
end 
figure(20);
imshow(xt); %Draw a new image
title('Fitted contour image');

%---Comparative Analysis of Lateral Fitting Curve Droplet Boundary Diagrams and Actual Diagrams---%
figure(21);
subplot(121),imshow(RGB1),title('Target comparison graph');
%The side fitting curve droplet boundary is marked in green on the original image to observe the differences.
hold on
for x =B(1,1,1):B(cha,1,1)  %61-82
    plot(C-r1(x-B(1,1,1)+1),x,'g.')   %Mark the right contour
    plot(C+r1(x-B(1,1,1)+1),x,'g.')   %Mark the left contour
end
for x =B(1,1,1):B(gd,1,1)  %61-158
    plot(C-r2(x-B(1,1,1)+1),x,'g.')         %Mark the right contour
    plot(C+r2(x-B(1,1,1)+1),x,'g.')         %Mark the left contour
end
for x=B(gd+1,1,1):B(p1,1,1) %159=214
    plot(C-r3(x-B(gd,1,1)),x,'g.')   %Mark the right contour
    plot(C+r3(x-B(gd,1,1)),x,'g.')   %Mark the left contour
end 

for x=C-r1(cha):C+r1(cha)
    plot(x,B(cha,1,1),'g.')  %Trace the bottom of the concave curve and seal it
end
for x=C-r3(p1-gd):C+r3(p1-gd)
    plot(x,B(p1,1,1),'g.')  %Trace the last line and seal it.
end 

subplot(122),imshow(RGB2),title('Background comparison graph');
%The side fitting curve droplet boundary is marked in green on the original image to observe the differences.
hold on
for x =B(1,1,1):B(cha,1,1)  %61-82
    plot(C-r1(x-B(1,1,1)+1),x,'g.')   %Mark the right contour
    plot(C+r1(x-B(1,1,1)+1),x,'g.')   %Mark the left contour
end
for x =B(1,1,1):B(gd,1,1)  %61-158
    plot(C-r2(x-B(1,1,1)+1),x,'g.')         %Mark the right contour
    plot(C+r2(x-B(1,1,1)+1),x,'g.')         %Mark the left contour
end
for x=B(gd+1,1,1):B(p1,1,1) %159=214
    plot(C-r3(x-B(gd,1,1)),x,'g.')   %Mark the right contour
    plot(C+r3(x-B(gd,1,1)),x,'g.')   %Mark the left contour
end 

for x=C-r1(cha):C+r1(cha)
    plot(x,B(cha,1,1),'g.')  %Trace the bottom of the concave curve and seal it
end
for x=C-r3(p1-gd):C+r3(p1-gd)
    plot(x,B(p1,1,1),'g.')  %Trace the last line and seal it
end 

figure(22);
imshow(BW),title('Target comparison graph');
%The side fitting curve droplet boundary is marked in green on the original image to observe the differences.
hold on
for x =B(1,1,1):B(cha,1,1)  %61-82
    plot(C-r1(x-B(1,1,1)+1),x,'g.')   %Mark the right contour
    plot(C+r1(x-B(1,1,1)+1),x,'g.')   %Mark the left contour
end
for x =B(1,1,1):B(gd,1,1)  %61-158
    plot(C-r2(x-B(1,1,1)+1),x,'g.')         %Mark the right contour
    plot(C+r2(x-B(1,1,1)+1),x,'g.')         %Mark the left contour
end
for x=B(gd+1,1,1):B(p1,1,1) %159=214
    plot(C-r3(x-B(gd,1,1)),x,'g.')   %Mark the right contour
    plot(C+r3(x-B(gd,1,1)),x,'g.')   %Mark the left contour
end 

for x=C-r1(cha):C+r1(cha)
    plot(x,B(cha,1,1),'g.')  %Trace the bottom of the concave curve and seal it
end
for x=C-r3(p1-gd):C+r3(p1-gd)
    plot(x,B(p1,1,1),'g.')  %Trace the last line and seal it
end 

% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%--Fill the new graph--%
xt1=xt;
for i=1:m                         %Specifically between 61 and 214                    
   for j=1:C                      %From the left boundary to the medial axis
      if xt1(i,j)==1&xt1(i,j+1)==0    
         xt1(i,j+1)=1; %Fill the left side
      elseif xt1(i,j)==1&xt1(i,j+1)==1
          break
      end
   end
   for j=n:-1:C                 %From the right boundary to the medial axis
      if xt1(i,j)==1&xt1(i,j-1)==0
         xt1(i,j-1)=1; %Fill the right side
      elseif xt1(i,j)==1&xt1(i,j-1)==1
          break
      end
   end
end 

%---Inverting changes the new image to resemble the original one (black droplets, white background)---%
xt1=1-xt1;
figure(23);
imshow(xt1); 
title('Fitted Filler image');

% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
A3=xt1;
[m,n]=size(A3);                    %%%%m=242,n=469
flag=0;  
exist=0; 
count=0; 
p=1;     
for i=1:m
    for j=1:n
        if A3(i,j)==0&count==0
            B3(p,1,1)=i;  %B(1,,1)=(28,231)
            B3(p,2,1)=j; %B(201,,1)=(228,214)
            count=count+1; 
            flag=1;        
            exist=1;
        end
        if A3(i,j)==1&count==1
            B3(p,1,2)=i;      %B(1,,2)=(28,241)
            B3(p,2,2)=j-1;  %B(201,,2)=(228,258)
            count=count+1; 
            break;  
        end
    end
    if flag==1 
        p=p+1;
        count=0;
        flag=0; 
    elseif flag==0&exist==1
        break; 
    end
end 
p3=p-1;   %Number of Fitted Graph Border Points, 153
pp3=p;

p4=0;
for i=1:p3
    if B3(i,2,2)>C
        p4=i-1;     %21
        break
    end
    i=i+1;
end

max3=0;
for i=1:p3
   R3(i)=B3(i,2,2)-B3(i,2,1);
   if R3(i)>max3
      max3=R3(i);
      t3=i;
   end
end


% % %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
% ---Solid figure of revolution---%
for i=1:201              %Here, only 1:201 can be written, not 1:p3, otherwise an error will occur.           
    rx(i)=B3(i,1,1);    
    ry(i)=B3(i,2,1);
end

for i=1:p3              %Moving the center of the ellipse to the coordinate axis 0 point.  
    rx(i)=B3(round(p3/2),1,1)-rx(i);     %Coordinate transformation, placing the contour of the right edge horizontally on the x-axis.      
    ry(i)=C-ry(i);
end
for i=1:p3    
    temp(i)=ry(202-i);   %Here, only 202-i can be written, not p3+1-i, otherwise an error will occur.  
end
ry=temp;
X=rx;
Y=ry;

[X,Y,Z]=cylinder(Y,30);    
Z=Z*p3;            
figure(24);
mesh(X,Y,Z);
title('Solid figure of revolution');

hold on
% ----------------------------------------------------------
for i=1:201              %Here, only 1:201 can be written, not 1:p4, otherwise an error will occur.           
    rx2(i)=B3(i,1,2);    
    ry2(i)=B3(i,2,2);
end

for i=1:p4              %Moving the center of the ellipse to the coordinate axis 0 point.  
    rx2(i)=rx2(i)-B3(round(p3/2),1,1);     %Coordinate transformation, placing the contour of the right edge horizontally on the x-axis.      
    ry2(i)=ry2(i)-C;
end
for i=1:p4    
    temp2(i)=ry2(202-i);   %Here, only 202-i can be written, not p4+1-i, otherwise an error will occur.  
end
ry2=temp2;
X2=rx2;
Y2=ry2;

[X2,Y2,Z2]=cylinder(Y2,10); 
Z2=Z2*p4;                                    
figure(24);
mesh(X2,Y2,Z2+p3-p4);
title('Solid figure of revolution');

figure(25);
ttl={'Main view','Top view'};
angle={[0,0],[0 90]};
for i=1:2   
    subplot(1,2,i);   
    mesh(X,Y,Z);  
    hold on
    mesh(X2,Y2,Z2+p3-p4);
    view(angle{i});title(ttl{i});
end
% %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%---Integrate for volume---%          
v=0;
syms x;

v1=sum(r1.^2)*pi;         
v2=sum(r2.^2)*pi;         
v3=sum(r3.^2)*pi;         

v=v2+v3-v1;
v=double(v);            
v=vpa(v,10);                        
disp('The pixel volume is');
disp(v);




            
               
               
               
               
               