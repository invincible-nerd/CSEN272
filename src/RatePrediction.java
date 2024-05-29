import java.util.*;

public class RatePrediction {
    //read train.txt and test.txt, write rate predictions as txt

    public static void main(String[] args) {
        Util util=new Util();

        List<int[]> train=util.read("D:\\SantaClaraU\\24spring\\IR\\Project2\\train.txt");
        //0 denotes unknown, to be predicted
        List<int[]> test5=util.read("D:\\SantaClaraU\\24spring\\IR\\Project2\\test5.txt");
        List<int[]> test10=util.read("D:\\SantaClaraU\\24spring\\IR\\Project2\\test10.txt");
        List<int[]> test20=util.read("D:\\SantaClaraU\\24spring\\IR\\Project2\\test20.txt");

        UserBasedCF userBasedCF=new UserBasedCF(10);
        List<int[]> res5 = userBasedCF.cosineSimilarity(train, test5, 200);
        List<int[]> res10 = userBasedCF.cosineSimilarity(train, test10, 300);
        List<int[]> res20 = userBasedCF.cosineSimilarity(train, test20, 400);
        util.write(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\res_5.txt");
        util.write(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\res_10.txt");
        util.write(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\res_20.txt");

//        UserBasedCF userBasedCF=new UserBasedCF(10);
//        List<int[]> res5 = userBasedCF.pearsonCorrelation(train, test5, 200);
//        List<int[]> res10 = userBasedCF.pearsonCorrelation(train, test10, 300);
//        List<int[]> res20 = userBasedCF.pearsonCorrelation(train, test20, 400);
//        util.write(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pres_5.txt");
//        util.write(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pres_10.txt");
//        util.write(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pres_20.txt");

//        UserBasedCF userBasedCF=new UserBasedCF(10);
//        List<int[]> res5 = userBasedCF.pearsonCorrelationIUF(train, test5, 200);
//        List<int[]> res10 = userBasedCF.pearsonCorrelationIUF(train, test10, 300);
//        List<int[]> res20 = userBasedCF.pearsonCorrelationIUF(train, test20, 400);
//        util.write(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pires_5.txt");
//        util.write(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pires_10.txt");
//        util.write(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pires_20.txt");

//        UserBasedCF userBasedCF=new UserBasedCF(10);
//        List<int[]> res5 = userBasedCF.pearsonCorrelationCA(train, test5, 200);
//        List<int[]> res10 = userBasedCF.pearsonCorrelationCA(train, test10, 300);
//        List<int[]> res20 = userBasedCF.pearsonCorrelationCA(train, test20, 400);
//        util.write(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pcres_5.txt");
//        util.write(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pcres_10.txt");
//        util.write(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\pcres_20.txt");

//        UserBasedCF userBasedCF=new UserBasedCF(10);
//        List<int[]> res5 = userBasedCF.pearsonCorrelationIUFCA(train, test5, 200);
//        List<int[]> res10 = userBasedCF.pearsonCorrelationIUFCA(train, test10, 300);
//        List<int[]> res20 = userBasedCF.pearsonCorrelationIUFCA(train, test20, 400);
//        util.write(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\picres_5.txt");
//        util.write(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\picres_10.txt");
//        util.write(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\picres_20.txt");

//        ItemBasedCF itemBasedCF1=new ItemBasedCF(5);
//        List<int[]> res5 = itemBasedCF1.adjustedCosineSimilarity(train, test5, 200);
//        ItemBasedCF itemBasedCF2=new ItemBasedCF(10);
//        List<int[]> res10 = itemBasedCF2.adjustedCosineSimilarity(train, test10, 300);
//        ItemBasedCF itemBasedCF3=new ItemBasedCF(20);
//        List<int[]> res20 = itemBasedCF3.adjustedCosineSimilarity(train, test20, 400);
//        util.write(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\ires_5.txt");
//        util.write(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\ires_10.txt");
//        util.write(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\ires_20.txt");

//        UserBasedCF userBasedCF=new UserBasedCF(10);
//        List<String[]> res5 = userBasedCF.customized(train, test5, 200);
//        List<String[]> res10 = userBasedCF.customized(train, test10, 300);
//        List<String[]> res20 = userBasedCF.customized(train, test20, 400);
//        util.writeS(res5,"D:\\SantaClaraU\\24spring\\IR\\Project2\\Cres_5.txt");
//        util.writeS(res10,"D:\\SantaClaraU\\24spring\\IR\\Project2\\Cres_10.txt");
//        util.writeS(res20,"D:\\SantaClaraU\\24spring\\IR\\Project2\\Cres_20.txt");
    }

//    public static void main(String[] args) {
//        PriorityQueue<Double> pq=new PriorityQueue<>((x,y)->(Double.compare(y,x)));
//        pq.offer(0.49004316653772806);
//        pq.offer(0.4365912105930143);
//        pq.offer(0.4248868360907387);
//        pq.offer(0.47912754646986505);
//        System.out.println(pq.poll());
//        System.out.println(pq.poll());
//        System.out.println(pq.poll());
//        System.out.println(pq.poll());
//    }
}
