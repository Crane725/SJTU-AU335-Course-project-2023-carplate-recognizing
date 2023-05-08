clc
clear
close all%���ó���
% 1����ͼ��Ԥ����
% 1.1 RGB���˺ͻҶȼ���
bar=waitbar(0,'Preprocessing in RGB filtering and bipolarizing...');
rgb_image=imread('test.jpg');%��ȡͼƬ
fR=rgb_image(:, :, 1);%��ȡ��ɫ����
fG=rgb_image(:, :, 2);%��ȡ��ɫ����
fB=rgb_image(:, :, 3);%��ȡ��ɫ����

figure(1),subplot(1,2,1),
imshow(rgb_image);                                                                                                                                                                                                                                                                                                 
title('Original');

for i=1:size(rgb_image,1)   %����ͼ���е�������ɫ���м���
    for j=1:size(rgb_image,2)
        if fR(i,j)>75|| fG(i,j)<30|| fG(i,j)>150|| fB(i,j)<120  %��������
            fR(i,j)=255;
            fG(i,j)=255;
            fB(i,j)=255;
        else
            fR(i,j)=0;
            fG(i,j)=0;
            fB(i,j)=0;
        end
    end
end
RGBtrans=cat(3,fR,fG,fB);
figure(1),subplot(1,2,2),
imshow(RGBtrans);                                                                                                                                                                                                                                                                                                 
title('RGB filtered & Bipolarized');
I=RGBtrans; %���뼫��ͼ��
I_gray=rgb2gray(I);

% 1.2���ƶ�λ
waitbar(0,bar,'Preprocessing in plate locating...');
[y,x]=size(I_gray);		%�����и����ֱ�������y,x��
Ysum=zeros(y,1);	%�������лҶ�ֵ�ܺ͵�����
Xsum=zeros(x,1);
for i=1:y   %�Ӻ����лҶ�ֵ�������ص��һ����
    for j=1:x
		Ysum(i)=Ysum(i)+double(I_gray(i,j));    %����ÿ�е����غ�
        Xsum(j)=Xsum(j)+double(I_gray(i,j));    %����ÿ�е����غ�
    end
end
[Ymin,Ypos]=min(Ysum);	%����������ֵ����Сֵ��
[Xmin,Xpos]=min(Xsum);  %����������ֵ����Сֵ
Ytop=Ypos;  %�ҳ����Ƶ��ϱ߽�
while((Ysum(Ytop)<=255*(x-5)) && (Ytop>1))
    Ytop=Ytop-1;
end

Ybottom=Ypos;   %�ҳ����Ƶ��±߽�
while((Ysum(Ybottom)<=255*(x-5)) && (Ybottom<y))
    Ybottom=Ybottom+1;
end

Xleft=Xpos;  %�ҳ����Ƶ���߽�
while((Xsum(Xleft)<=255*(y-5)) && (Xleft>1))
    Xleft=Xleft-1;
end

Xright=Xpos;   %�ҳ����Ƶ��ұ߽�
while((Xsum(Xright)<=255*(y-5)) && (Xright<x))
    Xright=Xright+1;
end



if 255*y-Xsum(Xleft+1)<=5000||255*x-Ysum(Ytop+1)<=3000
    filled=bwareaopen(I_gray,round(x*y/2));
    Ysum=zeros(y,1);	%�������лҶ�ֵ�ܺ͵�����
    Xsum=zeros(x,1);
    for i=1:y   %�Ӻ����лҶ�ֵ�������ص��һ����
        for j=1:x
		    Ysum(i)=Ysum(i)+double(filled(i,j))*255;    %����ÿ�е����غ�
            Xsum(j)=Xsum(j)+double(filled(i,j))*255;    %����ÿ�е����غ�
        end
    end
    Ysubtop=Ytop;
    Ysubbottom=Ybottom;
    Xsubleft=Xleft;
    Xsubright=Xright;
    for i=Ytop:Ybottom
        if Ysum(i)<=0.97*Ymin&&Ysubbottom==Ybottom
            Ysubbottom=Ybottom-i+Ytop;
        end
        if Ysum(i)<=0.97*Ymin&&Ysubtop==Ytop
            Ysubtop=i;
        end
        if Ysubbottom~=Ybottom&&Ysubtop~=Ytop
            break;
        end
    end
    for j=Xleft:Xright
        if Xsum(j)<=1.02*Xmin&&Xsubright==Xright
            Xsubright=Xright-j+Xleft;
        end
        if Xsum(j)<=1.02*Xmin&&Xsubleft==Xleft
            Xsubleft=j;
        end
        if Xsubright~=Xright&&Xsubleft~=Xleft
            break;
        end
    end
    if filled(Ytop,Xleft)==1&&filled(Ybottom,Xright)==1
        lt=[Xsubleft,Ysubtop];
        rt=[Xright,Ytop];
        lb=[Xleft,Ybottom];
        rb=[Xsubright,Ysubbottom];
    else
        if filled(Ybottom,Xleft)==1&&filled(Ytop,Xright)==1
            lt=[Xleft,Ytop];
            rt=[Xsubright,Ysubtop];
            lb=[Xsubleft,Ysubbottom];
            rb=[Xright,Ybottom];
        end
    end
    boxm=[lt;rt;lb;rb];
    boxf=[round(0.3175*x),round(0.5676*y);
    round(0.6867*x),round(0.5676*y);
    round(0.3175*x),round(0.7612*y);
    round(0.6867*x),round(0.7612*y)];
    TF = fitgeotrans(boxm,boxf,'projective');
    outview = imref2d(size(I_gray));
    I_gray = imwarp(I_gray,TF,'fillvalues',255,'outputview',outview);
    Ytop=round(0.5676*y);
    Ybottom=round(0.7612*y);
    Xleft=round(0.3175*x);
    Xright=round(0.6867*x);
end

Ytop=Ytop+20;
Ybottom=Ybottom-20;
Xleft=Xleft+20;
Xright=Xright-20;
dw=I_gray(Ytop:Ybottom,Xleft:Xright,:);
taka=Ybottom-Ytop+1;
yoko=Xright-Xleft+1;
Ytop=1;
Ybottom=taka;
Xleft=1;
Xright=yoko;
Ysum=zeros(taka,1);	%�������лҶ�ֵ�ܺ͵�����
Xsum=zeros(yoko,1);
for i=1:taka   %�Ӻ����лҶ�ֵ�������ص��һ����
    for j=1:yoko
		Ysum(i)=Ysum(i)+double(dw(i,j));    %����ÿ�е����غ�
        Xsum(j)=Xsum(j)+double(dw(i,j));    %����ÿ�е����غ�
    end
end
figure(2),subplot(1,2,1),
imshow(RGBtrans);                                                                                                                                                                                                                                                                                              
title('original');
figure(2),subplot(1,2,2),
imshow(dw);                                                                                                                                                                                                                                                                                                 
title('transformed & incised');
for i=1:taka
    if Ysum(i)<0.3*255*yoko&&Ysum(i)>0.17*255*yoko
        Ytop=i;
        break;
    end
end
for i=1:taka
    if Ysum(taka-i+1)<0.3*255*yoko&&Ysum(taka-i+1)>0.17*255*yoko
        Ybottom=taka-i+1;
        break;
    end
end
for j=1:yoko
    if Xsum(j)<0.2*255*taka
        Xleft=j;
        break;
    end
end
for j=1:yoko
    if Xsum(yoko-j+1)<0.2*255*taka
        Xright=yoko-j+1;
        break;
    end
end
dw=dw(Ytop:Ybottom,Xleft:Xright,:);
dw=imresize(dw,[200,811]);
se=strel('line',5,45);
dw=imerode(dw,se);
% 1.3 ȥ��

noiseinclude=dw;
taka=size(dw,1);
yoko=size(dw,2);
waitbar(0,bar,'  Preprocessing in noise eliminating...');
dw=imbinarize(dw);
%Ϊͼ����Ӱ���ͨ�ߣ�Ϊ����ɾ��С������׼��
for i=1:taka    %Ϊ��һλ���������ֱ��������ɾ��С����ʱ����ɾ
    dw(i,round(0.08*yoko))=1;
    dw(i,round(0.06*yoko))=1;
    dw(i,round(0.03*yoko))=1;
end
for j=1:round(0.15*yoko)    %Ϊ��һλ�������ˮƽ��������ɾ��С����ʱ����ɾ
    dw(round(0.5*taka),j)=1;
end

dw=bwareaopen(dw,round(8*taka)); %ɾ��С�������
%������������ȥ��Ϊ��ɾ��С�������ӵĺڰ���
for i=1:taka-1
    if dw(i+1,round(0.06*yoko))==1&&dw(i,round(0.06*yoko)-1)==0&&dw(i,round(0.06*yoko)+1)==0
        dw(i,round(0.06*yoko))=0;
    end
    if dw(i+1,round(0.08*yoko))==1&&dw(i,round(0.08*yoko)-1)==0&&dw(i,round(0.08*yoko)+1)==0
        dw(i,round(0.08*yoko))=0;
    end
    if dw(i+1,round(0.03*yoko))==1&&dw(i,round(0.03*yoko)-1)==0&&dw(i,round(0.03*yoko)+1)==0
        dw(i,round(0.03*yoko))=0;
    end
end
for j=1:round(0.15*yoko)
    dw(round(0.5*taka),j)=1;
    if dw(round(0.5*taka),j+1)==1&&dw(round(0.5*taka)+1,j)==0&&dw(round(0.5*taka)-1,j)==0
        dw(round(0.5*taka),j)=0;
    end
end
dw=bwareaopen(dw,100);    %�ٴ�ɾ��С����
dw=dw.*255;%����ֵͼ��תΪ�Ҷ�ֵ
figure(3),subplot(1,2,1),
imshow(noiseinclude);                                                                                                                                                                                                                                                                                              
title('original');
figure(3),subplot(1,2,2),
imshow(dw);                                                                                                                                                                                                                                                                                                 
title('noise eliminated');
                      

%������Ԫ�ָ�
waitbar(0,bar,'Preprocessing in words incising...');
%�����лҶ�ֵ�ܺ͵�����
Xaccum=zeros(yoko,1);
Yaccum=zeros(taka,1);
for i=1:taka   %�Ӻ��лҶ�ֵ�������ص��һ����
    for j=1:yoko
        Yaccum(i)=Yaccum(i)+double(dw(i,j))/255;
    end
end
headline=1;
bottomline=taka;
for i=1:taka
    if (Yaccum(i)>=220)&&headline==1
        headline=i;
    end
    if (Yaccum(taka-i+1)>=220)&&bottomline==taka
        bottomline=taka-i+1;
    end
    if headline~=1&&bottomline~=taka
        break;
    end
end
dw=dw(headline:bottomline,:,:);
taka=bottomline-headline+1;
for i=1:taka   %�Ӻ��лҶ�ֵ�������ص��һ����
    for j=1:yoko
        Xaccum(j)=Xaccum(j)+double(dw(i,j))/255;
    end
end

% ��ʼ����Ҫ����
hori=zeros(14,1);
tmp=1;
last=1;     %��last����¼�ϴβü��ķ�λ
for j=1:yoko-3
    if tmp==15
            break;
    end
    if  Xaccum(j)~=0&&Xaccum(j+1)~=0&&Xaccum(j+2)~=0&&Xaccum(j+3)~=0&&last==1
            hori(tmp)=j;
            last=last*-1;
            tmp=tmp+1;
            continue;
    end
    if  Xaccum(j)==0&&Xaccum(j+1)==0&&Xaccum(j+2)==0&&Xaccum(j+3)==0&&last==-1&&j-hori(tmp-1)>=70
            hori(tmp)=j;
            last=last*-1;
            tmp=tmp+1;
    end
    if j==yoko-3&&hori(14)==0
        hori(14)=yoko;
    end
end
file_path='.\samples\';
plate=['i','i','i','i','i','i','i'];
class=['location1\';'location2\';'sequences\';'sequences\';'sequences\';'sequences\';'sequences\'];
error=0;
for chara=1:7
    if error~=0
        break;
    end
    score=-inf;
    wordchara=dw(:,round(hori(2*chara-1)):round(hori(2*chara)),:);
    figure(4),subplot(2,7,chara),imshow(wordchara);
    for k=3:size(dir([file_path,class(chara,:)]))
        processing=sprintf('Words recognizing ... %.1f %s',100*(k-2+31*(chara>1)+24*(chara>2)+34*(chara-3)*(chara>3))/225,'%');
        waitbar(((k-2+31*(chara>1)+24*(chara>2)+34*(chara-3)*(chara>3)))/225,bar,processing);
        file=dir([file_path,class(chara,:)]);
        file_name=file(k).name;
        sample=imread([[file_path,class(chara,:)],file_name]);
        sample=imresize(sample,[size(wordchara,1),size(wordchara,2)]);
        tmpscore=0;
        for i=1:size(sample,1)   %����ͼ���е�������ɫ���м���
            for j=1:size(sample,2)
                if sample(i,j,1)>127  %��������
                    sample(i,j,1)=255;
                else
                    sample(i,j,1)=0;
                end
            end
        end
        for i=1:size(wordchara,1)
            for j=1:size(wordchara,2)
                if sample(i,j)==wordchara(i,j)&&wordchara(i,j)==255
                    tmpscore=tmpscore+1;
                else
                    if sample(i,j)==wordchara(i,j)&&wordchara(i,j)==0
                        tmpscore=tmpscore+1;
                    else
                        tmpscore=tmpscore-1;
                    end
                end
            end
        end
        if tmpscore>score
            score=tmpscore;
            tmpplate=file_name;
        else
            if tmpscore==score
                error=chara;
                break;
            else
                continue;
            end
        end
    end
    figure(4),subplot(2,7,chara+7),imshow(imread([[file_path,class(chara,:)],tmpplate]));
    title(erase(tmpplate,".jpg"),'FontSize',20,'Color',[1,1,1],'EdgeColor',[100/255,150/255,170/255],'BackgroundColor',[0,80/255,170/255],'FontWeight','bold');
    tmpplate=erase(tmpplate,".jpg");
    plate(chara)=tmpplate;
end
if error==0
    waitbar(1,bar,'Running Over');
    plate=[plate(1:2),'��',plate(3:7)];
    sprintf(plate)
else
    waitbar(0,bar,sprintf('Equivalent ERROR occurred in word %d',error));
end