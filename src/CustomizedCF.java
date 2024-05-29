import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

public class CustomizedCF {
    private class W{
        int userId;
        double weight;
        W(int userId, double weight){
            this.userId=userId;
            this.weight=weight;
        }
    }

    CustomizedCF(){}

    public List<int[]> customized(List<int[]> train_list, List<int[]> test_list, int d){
        int K=10;
        switch(d){
            case 200:
                K=15;
                break;
            case 300:
                K=30;
                break;
            case 400:
                K=45;
                break;
        }
        int[][] train=new int[201][1001];
        int[][] test=new int[101][1001];
        double[][] cos=new double[101][201]; //cosine similarity of user i+d, j

        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
        }
        for(int[] arr : test_list){
            if(arr[2]==0){
                continue;
            }
            int u=arr[0]-d;
            int i=arr[1];
            test[u][i]=arr[2];
        }


        for(int u1=1; u1<=100; u1++){
            //cosine = c/(sqrt(a)*sqrt(b))
            //only consider when both u1,i and u2,i exist


            for(int u2=1; u2<=200; u2++){
                int a=0; //u1 test
                int b=0; //u2 train
                int c=0;
                for(int i=1; i<=1000; i++){
                    if(test[u1][i]!=0 && train[u2][i]!=0){
                        a+=test[u1][i]*test[u1][i];
                        b+=train[u2][i]*train[u2][i];
                        c+=test[u1][i]*train[u2][i];
                    }
                }
                if(a!=0){
                    cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(b));
                }
                //qs[u1].offer(new W(u2,cos[u1][u2]));

            }
        }
        List<int[]> res=new ArrayList<>();
        for(int[] arr : test_list){
            if(arr[2]==0){
                int u1=arr[0]-d;
                int i=arr[1];
                double sum=0;
                double totalWeights=0;
                PriorityQueue<W> pq=new PriorityQueue<>((x, y)->(Double.compare(y.weight,x.weight)));
                for(int u2=1; u2<=200; u2++){
                    pq.offer(new W(u2,cos[u1][u2]));
                }
                int k=0;
                while(k<K && !pq.isEmpty()){
                    W w=pq.poll();
                    int u2=w.userId;
                    double weight=w.weight;
                    if(train[u2][i]!=0){
                        k++;
                        sum+=weight*train[u2][i];
                        totalWeights+=weight;
                    }
                }
                double pd=sum/totalWeights;
                int p=5;

                if(pd<1.5){
                    p=1;
                }else if(pd<2.5){
                    p=2;
                }else if(pd<3.5){
                    p=3;
                }else if(pd<4.5){
                    p=4;
                }
                res.add(new int[]{u1,i,p});
            }
        }
        return res;
    }
}
