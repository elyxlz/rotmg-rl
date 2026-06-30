#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <arpa/inet.h>
static int (*real_connect)(int,const struct sockaddr*,socklen_t)=0;
int connect(int fd,const struct sockaddr *a,socklen_t l){
  if(!real_connect) real_connect=dlsym(RTLD_NEXT,"connect");
  if(a && a->sa_family==AF_INET){
    struct sockaddr_in *in=(struct sockaddr_in*)a;
    if(ntohs(in->sin_port)==2050){
      struct sockaddr_in c=*in; c.sin_port=htons(2052);
      return real_connect(fd,(struct sockaddr*)&c,l);
    }
  }
  return real_connect(fd,a,l);
}
