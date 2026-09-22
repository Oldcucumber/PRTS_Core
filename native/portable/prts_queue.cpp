#include "prts_queue.h"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());size_t n=v.size();
    return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;
}
struct Quad {
    double a,b,c,d,e,f,g,h;
    explicit Quad(const float *q) {
        double dx1=q[2]-q[4],dx2=q[6]-q[4],dx3=q[0]-q[2]+q[4]-q[6];
        double dy1=q[3]-q[5],dy2=q[7]-q[5],dy3=q[1]-q[3]+q[5]-q[7];
        double denominator=dx1*dy2-dx2*dy1;g=0;h=0;
        if(std::abs(denominator)>1e-9) {
            g=(dx3*dy2-dx2*dy3)/denominator;h=(dx1*dy3-dx3*dy1)/denominator;
        }
        a=q[2]-q[0]+g*q[2];b=q[6]-q[0]+h*q[6];c=q[0];
        d=q[3]-q[1]+g*q[3];e=q[7]-q[1]+h*q[7];f=q[1];
    }
    std::pair<double,double> at(double u,double v) const {
        double t=g*u+h*v+1;
        return {(a*u+b*v+c)/t,(d*u+e*v+f)/t};
    }
};
}
int32_t prts_queue_split_rgb(const uint8_t *rgb,int32_t iw,int32_t ih,const float *q,
    int32_t count,int32_t *ranges,float *boxes) {
    if(!rgb||!q||!ranges||!boxes||iw<1||ih<1||count<1||count>1024)return -1;
    ranges[0]=0;ranges[1]=count;std::copy(q,q+8,boxes);
    int w=int(std::nearbyint(std::hypot(q[2]-q[0],q[3]-q[1])));
    int h=int(std::nearbyint(std::hypot(q[6]-q[0],q[7]-q[1])));
    if(count<3||w<8||h<8)return 1;
    if(w>16384||h>16384)return -1;
    Quad transform(q);std::vector<uint8_t> gray(size_t(w)*h);double histogram[256]={};
    auto pixel=[&](int x,int y,int c)->double {
        return x>=0&&y>=0&&x<iw&&y<ih?rgb[(size_t(y)*iw+x)*3+c]:0;
    };
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        auto p=transform.at(double(x)/(w-1),double(y)/(h-1));
        double sx=std::nearbyint(p.first*32)/32,sy=std::nearbyint(p.second*32)/32;
        int ix=int(std::floor(sx)),iy=int(std::floor(sy));double fx=sx-ix,fy=sy-iy;
        int color[3];
        for(int c=0;c<3;++c)color[c]=int(std::floor(
            (pixel(ix,iy,c)*(1-fx)+pixel(ix+1,iy,c)*fx)*(1-fy)+
            (pixel(ix,iy+1,c)*(1-fx)+pixel(ix+1,iy+1,c)*fx)*fy+.5));
        int value=(color[0]*4899+color[1]*9617+color[2]*1868+8192)>>14;
        gray[size_t(y)*w+x]=uint8_t(value);histogram[value]+=1;
    }
    double total=double(w)*h,sum=0,background=0,sumBackground=0,best=-1;int threshold=0;
    for(int i=0;i<256;++i)sum+=i*histogram[i];
    for(int i=0;i<256;++i) {
        background+=histogram[i];sumBackground+=i*histogram[i];
        if(background==0||background==total)continue;
        double difference=sumBackground/background-(sum-sumBackground)/(total-background);
        double variance=background*(total-background)*difference*difference;
        if(variance>best){best=variance;threshold=i;}
    }
    int bright=0;
    for(int x=0;x<w;++x)bright+=(gray[x]>threshold)+(gray[size_t(h-1)*w+x]>threshold);
    for(int y=0;y<h;++y)bright+=(gray[size_t(y)*w]>threshold)+(gray[size_t(y)*w+w-1]>threshold);
    bool inverted=double(bright)/(2*w+2*h)>127./255;
    std::vector<std::pair<int,int>> spans;int start=-1;
    for(int x=0;x<=w;++x) {
        int ink=0;
        if(x<w)for(int y=h/5;y<4*h/5;++y)ink+=((gray[size_t(y)*w+x]>threshold)!=inverted);
        bool occupied=x<w&&ink>.10*h;
        if(occupied&&start<0)start=x;
        if(!occupied&&start>=0) {
            if(x-start>=std::max(2.,h*.025))spans.push_back({start,x});
            start=-1;
        }
    }
    if(int(spans.size())!=count)return 1;
    std::vector<double> gaps,widths;
    for(int i=0;i<count;++i) {
        widths.push_back(spans[i].second-spans[i].first);
        if(i)gaps.push_back(spans[i].first-spans[i-1].second);
    }
    double gapThreshold=std::max(.6*median(widths),2*median(gaps));
    std::vector<int> boundaries={0};
    for(int i=0;i<count-1;++i)if(gaps[i]>gapThreshold)boundaries.push_back(i+1);
    if(boundaries.size()==1)return 1;
    boundaries.push_back(count);
    for(size_t i=0;i+1<boundaries.size();++i) {
        int a=boundaries[i],b=boundaries[i+1];ranges[2*i]=a;ranges[2*i+1]=b;
        double left=double(spans[a].first)/(w-1),right=double(spans[b-1].second)/(w-1);
        auto tl=transform.at(left,0),tr=transform.at(right,0),br=transform.at(right,1),bl=transform.at(left,1);
        double points[]={tl.first,tl.second,tr.first,tr.second,br.first,br.second,bl.first,bl.second};
        std::copy(points,points+8,boxes+8*i);
    }
    return int(boundaries.size())-1;
}
