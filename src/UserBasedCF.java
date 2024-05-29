import java.util.*;

public class UserBasedCF {
    private int K;
    UserBasedCF(int K){
        this.K=K;
    }

    private class W{
        int userId;
        double weight;
        W(int userId, double weight){
            this.userId=userId;
            this.weight=weight;
        }
    }

    private class W2{
        int userId;
        double weight;
        double abs;
        W2(int userId, double weight){
            this.userId=userId;
            this.weight=weight;
            this.abs=Math.abs(weight);
        }
    }

    //train: 1-200
    //test5: 201-300, d=200
    //test10: 301-400, d=300
    //test20: 401-500, d=400

    //movie: 1-1000

    // ONLY know the knowledge of the training data and the existing ratings for this user

    //precompute similarity weight of each user
    //for each unknown rating(u,i)
    //find k neighbors, who has a rating on movie i
    //calculate weighted average rate of neighbors as the predicted rating(u,i)
    public List<int[]> cosineSimilarity(List<int[]> train_list, List<int[]> test_list, int d){
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
        int[] B=new int[201];
        for(int u2=1; u2<=200; u2++){
            int b=0;
            for(int i=1; i<=1000; i++){
                if(train[u2][i]!=0){
                    b+=train[u2][i]*train[u2][i];
                }
            }
            B[u2]=b;
        }
        for(int u1=1; u1<=100; u1++){
            //cosine = c/(sqrt(a)*sqrt(b))
            int a=0;
            for(int i=1; i<=1000; i++){
                if(train[u1][i]!=0){
                    a+=train[u1][i]*train[u1][i];
                }
            }
            //all weights of (u1,u) in descending order
            //qs[u1]=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));

            for(int u2=1; u2<=200; u2++){
                int c=0;
                for(int i=1; i<=1000; i++){
                    if(train[u1][i]!=0 && train[u2][i]!=0){
                        c+=train[u1][i]*train[u2][i];
                    }
                }
                cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(B[u2]));
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
                PriorityQueue<W> pq=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));
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

    public List<int[]> pearsonCorrelation(List<int[]> train_list, List<int[]> test_list, int d){
        int[][] train=new int[201][1001];
        int[][] test=new int[101][1001];
        int[] trainSum=new int[201];
        int[] trainCount=new int[201];
        int[] testSum=new int[101];
        int[] testCount=new int[101];
        double[] trainAvg=new double[201]; //average rating of train user u2
        double[] testAvg=new double[101]; //average rating of test user u1
        double[][] cos=new double[101][201]; //pearson correlation of user i+d, j

        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
            trainSum[u]+=arr[2];
            trainCount[u]++;
        }
        for(int[] arr : test_list){
            if(arr[2]==0){
                continue;
            }
            int u=arr[0]-d;
            int i=arr[1];
            test[u][i]=arr[2];
            testSum[u]+=arr[2];
            testCount[u]++;
        }

        for(int u2=1; u2<=200; u2++){
            trainAvg[u2]=(double)trainSum[u2]/trainCount[u2];
        }
        for(int u1=1; u1<=100; u1++){
            testAvg[u1]=(double)testSum[u1]/testCount[u1];
        }

        double[] B=new double[201];
        for(int u2=1; u2<=200; u2++){
            double b=0;
            for(int i=1; i<=1000; i++){
                if(train[u2][i]!=0){
                    b+=(train[u2][i]-trainAvg[u2])*(train[u2][i]-trainAvg[u2]);
                }
            }
            B[u2]=b;
        }
        for(int u1=1; u1<=100; u1++){
            //cosine = c/(sqrt(a)*sqrt(b))
            double a=0;
            for(int i=1; i<=1000; i++){
                if(train[u1][i]!=0){
                    a+=(train[u1][i]-testAvg[u1])*(train[u1][i]-testAvg[u1]);
                }
            }
            //all weights of (u1,u) in descending order
            //qs[u1]=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));

            for(int u2=1; u2<=200; u2++){
                double c=0;
                for(int i=1; i<=1000; i++){
                    if(train[u1][i]!=0 && train[u2][i]!=0){
                        c+=(train[u1][i]-testAvg[u1])*(train[u2][i]-trainAvg[u2]);
                    }
                }
                cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(B[u2]));
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
                PriorityQueue<W2> pq=new PriorityQueue<>((x,y)->(Double.compare(y.abs,x.abs)));
                for(int u2=1; u2<=200; u2++){
                    pq.offer(new W2(u2,cos[u1][u2]));
                }
                int k=0;
                while(k<K && !pq.isEmpty()){
                    W2 w=pq.poll();
                    int u2=w.userId;
                    double weight=w.weight;
                    double abs=w.abs;
                    if(train[u2][i]!=0){
                        k++;
                        sum+=weight*(train[u2][i]-trainAvg[u2]);
                        totalWeights+=abs;
                    }
                }
                double pd=testAvg[u1]+sum/totalWeights;
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

    public List<int[]> pearsonCorrelationIUF(List<int[]> train_list, List<int[]> test_list, int d){
        int[][] train=new int[201][1001];
        int[][] test=new int[101][1001];
        int[] trainSum=new int[201];
        int[] trainCount=new int[201];
        int[] testSum=new int[101];
        int[] testCount=new int[101];
        double[] trainAvg=new double[201]; //average rating of train user u2
        double[] testAvg=new double[101]; //average rating of test user u1
        double[][] cos=new double[101][201]; //pearson correlation of user i+d, j

        int[] movieCount=new int[1001];
        double[] iuf=new double[1001];

        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
            trainSum[u]+=arr[2];
            trainCount[u]++;
            movieCount[i]++;
        }
        for(int[] arr : test_list){
            if(arr[2]==0){
                continue;
            }
            int u=arr[0]-d;
            int i=arr[1];
            test[u][i]=arr[2];
            testSum[u]+=arr[2];
            testCount[u]++;
        }

        for(int u2=1; u2<=200; u2++){
            trainAvg[u2]=(double)trainSum[u2]/trainCount[u2];
        }
        for(int u1=1; u1<=100; u1++){
            testAvg[u1]=(double)testSum[u1]/testCount[u1];
        }

        // penalize universally liked movies
        for(int i=1; i<=1000; i++){
            if(movieCount[i]!=0){
                iuf[i]=Math.log(200/movieCount[i]);
            }
        }

        double[] B=new double[201];
        for(int u2=1; u2<=200; u2++){
            double b=0;
            for(int i=1; i<=1000; i++){
                if(train[u2][i]!=0){
                    b+=(train[u2][i]-trainAvg[u2])*(train[u2][i]-trainAvg[u2])*iuf[i];
                }
            }
            B[u2]=b;
        }
        for(int u1=1; u1<=100; u1++){
            //cosine = c/(sqrt(a)*sqrt(b))
            double a=0;
            for(int i=1; i<=1000; i++){
                if(train[u1][i]!=0){
                    a+=(train[u1][i]-testAvg[u1])*(train[u1][i]-testAvg[u1])*iuf[i];
                }
            }
            //all weights of (u1,u) in descending order
            //qs[u1]=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));

            for(int u2=1; u2<=200; u2++){
                double c=0;
                for(int i=1; i<=1000; i++){
                    if(train[u1][i]!=0 && train[u2][i]!=0){
                        c+=(train[u1][i]-testAvg[u1])*(train[u2][i]-trainAvg[u2])*iuf[i];
                    }
                }
                cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(B[u2]));
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
                PriorityQueue<W2> pq=new PriorityQueue<>((x,y)->(Double.compare(y.abs,x.abs)));
                for(int u2=1; u2<=200; u2++){
                    pq.offer(new W2(u2,cos[u1][u2]));
                }
                int k=0;
                while(k<K && !pq.isEmpty()){
                    W2 w=pq.poll();
                    int u2=w.userId;
                    double weight=w.weight;
                    double abs=w.abs;
                    if(train[u2][i]!=0){
                        k++;
                        sum+=weight*(train[u2][i]-trainAvg[u2]);
                        totalWeights+=abs;
                    }
                }
                double pd=testAvg[u1]+sum/totalWeights;
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

    public List<int[]> pearsonCorrelationCA(List<int[]> train_list, List<int[]> test_list, int d){
        int[][] train=new int[201][1001];
        int[][] test=new int[101][1001];
        int[] trainSum=new int[201];
        int[] trainCount=new int[201];
        int[] testSum=new int[101];
        int[] testCount=new int[101];
        double[] trainAvg=new double[201]; //average rating of train user u2
        double[] testAvg=new double[101]; //average rating of test user u1
        double[][] cos=new double[101][201]; //pearson correlation of user i+d, j


        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
            trainSum[u]+=arr[2];
            trainCount[u]++;
        }
        for(int[] arr : test_list){
            if(arr[2]==0){
                continue;
            }
            int u=arr[0]-d;
            int i=arr[1];
            test[u][i]=arr[2];
            testSum[u]+=arr[2];
            testCount[u]++;
        }

        for(int u2=1; u2<=200; u2++){
            trainAvg[u2]=(double)trainSum[u2]/trainCount[u2];
        }
        for(int u1=1; u1<=100; u1++){
            testAvg[u1]=(double)testSum[u1]/testCount[u1];
        }


        double[] B=new double[201];
        for(int u2=1; u2<=200; u2++){
            double b=0;
            for(int i=1; i<=1000; i++){
                if(train[u2][i]!=0){
                    b+=(train[u2][i]-trainAvg[u2])*(train[u2][i]-trainAvg[u2]);
                }
            }
            B[u2]=b;
        }
        for(int u1=1; u1<=100; u1++){
            //cosine = c/(sqrt(a)*sqrt(b))
            double a=0;
            for(int i=1; i<=1000; i++){
                if(train[u1][i]!=0){
                    a+=(train[u1][i]-testAvg[u1])*(train[u1][i]-testAvg[u1]);
                }
            }
            //all weights of (u1,u) in descending order
            //qs[u1]=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));

            for(int u2=1; u2<=200; u2++){
                double c=0;
                for(int i=1; i<=1000; i++){
                    if(train[u1][i]!=0 && train[u2][i]!=0){
                        c+=(train[u1][i]-testAvg[u1])*(train[u2][i]-trainAvg[u2]);
                    }
                }
                cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(B[u2]));
            }
        }
        List<int[]> res=new ArrayList<>();
        for(int[] arr : test_list){
            if(arr[2]==0){
                int u1=arr[0]-d;
                int i=arr[1];
                double sum=0;
                double totalWeights=0;
                PriorityQueue<W2> pq=new PriorityQueue<>((x,y)->(Double.compare(y.abs,x.abs)));
                for(int u2=1; u2<=200; u2++){
                    //case amplification, p=2.5
                    pq.offer(new W2(u2,cos[u1][u2]*Math.pow(Math.abs(cos[u1][u2]),1.5)));
                }
                int k=0;
                while(k<K && !pq.isEmpty()){
                    W2 w=pq.poll();
                    int u2=w.userId;
                    double weight=w.weight;
                    double abs=w.abs;
                    if(train[u2][i]!=0){
                        k++;
                        sum+=weight*(train[u2][i]-trainAvg[u2]);
                        totalWeights+=abs;
                    }
                }
                double pd=testAvg[u1]+sum/totalWeights;
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

    public List<int[]> pearsonCorrelationIUFCA(List<int[]> train_list, List<int[]> test_list, int d){
        int[][] train=new int[201][1001];
        int[][] test=new int[101][1001];
        int[] trainSum=new int[201];
        int[] trainCount=new int[201];
        int[] testSum=new int[101];
        int[] testCount=new int[101];
        double[] trainAvg=new double[201]; //average rating of train user u2
        double[] testAvg=new double[101]; //average rating of test user u1
        double[][] cos=new double[101][201]; //pearson correlation of user i+d, j

        int[] movieCount=new int[1001];
        double[] iuf=new double[1001];

        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
            trainSum[u]+=arr[2];
            trainCount[u]++;
        }
        for(int[] arr : test_list){
            if(arr[2]==0){
                continue;
            }
            int u=arr[0]-d;
            int i=arr[1];
            test[u][i]=arr[2];
            testSum[u]+=arr[2];
            testCount[u]++;
        }

        for(int u2=1; u2<=200; u2++){
            trainAvg[u2]=(double)trainSum[u2]/trainCount[u2];
        }
        for(int u1=1; u1<=100; u1++){
            testAvg[u1]=(double)testSum[u1]/testCount[u1];
        }

        // penalize universally liked movies
        for(int i=1; i<=1000; i++){
            if(movieCount[i]!=0){
                iuf[i]=Math.log(200/movieCount[i]);
            }
        }


        double[] B=new double[201];
        for(int u2=1; u2<=200; u2++){
            double b=0;
            for(int i=1; i<=1000; i++){
                if(train[u2][i]!=0){
                    b+=(train[u2][i]-trainAvg[u2])*(train[u2][i]-trainAvg[u2])*iuf[i];
                }
            }
            B[u2]=b;
        }
        for(int u1=1; u1<=100; u1++){
            //cosine = c/(sqrt(a)*sqrt(b))
            double a=0;
            for(int i=1; i<=1000; i++){
                if(train[u1][i]!=0){
                    a+=(train[u1][i]-testAvg[u1])*(train[u1][i]-testAvg[u1])*iuf[i];
                }
            }
            //all weights of (u1,u) in descending order
            //qs[u1]=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));

            for(int u2=1; u2<=200; u2++){
                double c=0;
                for(int i=1; i<=1000; i++){
                    if(train[u1][i]!=0 && train[u2][i]!=0){
                        c+=(train[u1][i]-testAvg[u1])*(train[u2][i]-trainAvg[u2])*iuf[i];
                    }
                }
                cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(B[u2]));
            }
        }
        List<int[]> res=new ArrayList<>();
        for(int[] arr : test_list){
            if(arr[2]==0){
                int u1=arr[0]-d;
                int i=arr[1];
                double sum=0;
                double totalWeights=0;
                PriorityQueue<W2> pq=new PriorityQueue<>((x,y)->(Double.compare(y.abs,x.abs)));
                for(int u2=1; u2<=200; u2++){
                    //case amplification, p=2.5
                    pq.offer(new W2(u2,cos[u1][u2]*Math.pow(Math.abs(cos[u1][u2]),1.5)));
                }
                int k=0;
                while(k<K && !pq.isEmpty()){
                    W2 w=pq.poll();
                    int u2=w.userId;
                    double weight=w.weight;
                    double abs=w.abs;
                    if(train[u2][i]!=0){
                        k++;
                        sum+=weight*(train[u2][i]-trainAvg[u2]);
                        totalWeights+=abs;
                    }
                }
                double pd=testAvg[u1]+sum/totalWeights;
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

//    public List<String[]> customized(List<int[]> train_list, List<int[]> test_list, int d){
//        int[][] train=new int[201][1001];
//        int[][] test=new int[101][1001];
//        double[][] cos=new double[101][201]; //cosine similarity of user i+d, j
//
//        for(int[] arr : train_list){
//            int u=arr[0];
//            int i=arr[1];
//            train[u][i]=arr[2];
//        }
//        for(int[] arr : test_list){
//            if(arr[2]==0){
//                continue;
//            }
//            int u=arr[0]-d;
//            int i=arr[1];
//            test[u][i]=arr[2];
//        }
//        int[] B=new int[201];
//        for(int u2=1; u2<=200; u2++){
//            int b=0;
//            for(int i=1; i<=1000; i++){
//                if(train[u2][i]!=0){
//                    b+=train[u2][i]*train[u2][i];
//                }
//            }
//            B[u2]=b;
//        }
//        for(int u1=1; u1<=100; u1++){
//            //cosine = c/(sqrt(a)*sqrt(b))
//            int a=0;
//            for(int i=1; i<=1000; i++){
//                if(train[u1][i]!=0){
//                    a+=train[u1][i]*train[u1][i];
//                }
//            }
//            //all weights of (u1,u) in descending order
//            //qs[u1]=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));
//
//            for(int u2=1; u2<=200; u2++){
//                int c=0;
//                for(int i=1; i<=1000; i++){
//                    if(train[u1][i]!=0 && train[u2][i]!=0){
//                        c+=train[u1][i]*train[u2][i];
//                    }
//                }
//                cos[u1][u2]=c/(Math.sqrt(a)*Math.sqrt(B[u2]));
//                //qs[u1].offer(new W(u2,cos[u1][u2]));
//
//            }
//        }
//        List<String[]> res=new ArrayList<>();
//        for(int[] arr : test_list){
//            if(arr[2]==0){
//                int u1=arr[0]-d;
//                int i=arr[1];
//                double sum=0;
//                double totalWeights=0;
//                PriorityQueue<W> pq=new PriorityQueue<>((x,y)->(Double.compare(y.weight,x.weight)));
//                for(int u2=1; u2<=200; u2++){
//                    pq.offer(new W(u2,cos[u1][u2]));
//                }
//                int k=0;
//                while(k<K && !pq.isEmpty()){
//                    W w=pq.poll();
//                    int u2=w.userId;
//                    double weight=w.weight;
//                    if(train[u2][i]!=0){
//                        k++;
//                        sum+=weight*train[u2][i];
//                        totalWeights+=weight;
//                    }
//                }
//                double pd=sum/totalWeights;
//
//
//
//                String[] s=new String[3];
//                s[0]=String.valueOf(arr[0]);
//                s[1]=String.valueOf(arr[1]);
//                s[2]=String.valueOf(pd);
//                res.add(s);
//            }
//        }
//        return res;
//    }

}
