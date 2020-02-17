//VGG
import layer._
object VGG16{
  def loadParameter(){
    val file = scala.io.Source.fromFile("/home/share/vgg16_weights.txt").getLines
    var data = List[Array[Float]]()
    while(file.hasNext){
      val line = file.next
      if(line.trim.split(" ")(0) == "DATASET"){
        //ToParameter
        //delete
        file.next
        file.next
        file.next
        var param = List[String]()
        var target = file.next.trim
        while(target != "}"){
          param ::= target.split(":")(1)
          target = file.next.trim
        }
        data ::= param.reverse.mkString("").split(",").map(_.toFloat)
      }
    }
    
    val temp = data.reverse.toArray
    val newData = List(temp(0),temp(1),temp(20),temp(21),temp(28),temp(29),temp(30),
      temp(31),temp(2),temp(3),temp(4),temp(5),temp(6),temp(7),temp(8),temp(9),
      temp(10),temp(11),temp(12),temp(13),temp(14),temp(15),temp(16),temp(17),
      temp(18),temp(19),temp(22),temp(23),temp(24),temp(25),temp(26),temp(27))
    //save
    for(i<-0 until newData.size){
      val save = new java.io.PrintWriter("VGG16/VGG16Parameter"+(i+1))
      for(j<-0 until newData(i).size){
        save.println(newData(i)(j))
      }
      save.close
    }
  }//loadParameter

  //画像前処理
/*  def loadVGG16data(dir:String)={
    def fd(line:String) = line.split(",").map(_.toFloat).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val test_d = scala.io.Source.fromFile(dir+"/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir+"/test_t.txt").getLines.map(ft).toArray.head
    test_d.map(encode(_)).zip(test_t)
  }//loadVGG16Data
 */

  def encodeVGG16(V:Array[Array[Array[Float]]])={
    var R = List[Float]()
    var G = List[Float]()
    var B = List[Float]()
    for(i<-0 until V.size){
      for(j<-0 until V(0).size){
        R ::= V(i)(j)(0) - 103.939f
        G ::= V(i)(j)(1) - 116.779f
        B ::= V(i)(j)(2) - 123.68f
      }
    }
      (B.reverse ++ G.reverse ++ R.reverse).toArray
  }//encodeVGG16


  def main(){
    //----------network-----------------
    val N = new network()
    val layer = List(
      //convolution:KW,IH,IW,IC,OC
      //Pooling:BW=2,IC,IH,IW
      new Convolution(3,224,224,3,64),new ReLU(),
      new Convolution(3,224,224,64,64),new ReLU(),
      new Pooling(2,64,224,224),
      new Convolution(3,112,112,64,128),new ReLU(),
      new Convolution(3,112,112,128,128),new ReLU(),
      new Pooling(2,128,112,112),
      new Convolution(3,56,56,128,256),new ReLU(),
      new Convolution(3,56,56,256,256),new ReLU(),
      new Convolution(3,56,56,256,256),new ReLU(),
      new Pooling(2,256,56,56),
      new Convolution(3,28,28,256,512),new ReLU(),
      new Convolution(3,28,28,512,512),new ReLU(),
      new Convolution(3,28,28,512,512),new ReLU(),
      new Pooling(2,512,28,28),
      new Convolution(3,14,14,512,512),new ReLU(),
      new Convolution(3,14,14,512,512),new ReLU(),
      new Convolution(3,14,14,512,512),new ReLU(),
      new Pooling(2,512,14,14),
      new Affine(25088,4096),new ReLU(),
      new Affine(4096,4096),new ReLU(),
      new Affine(4096,1000)
    )

    //-----loadParameter()-----
    //convolution
    layer(0).load("VGG16/VGG16Parameter1")
    layer(2).load("VGG16/VGG16Parameter3")
    layer(5).load("VGG16/VGG16Parameter5")
    layer(7).load("VGG16/VGG16Parameter7")
    layer(10).load("VGG16/VGG16Parameter9")
    layer(12).load("VGG16/VGG16Parameter11")
    layer(14).load("VGG16/VGG16Parameter13")
    layer(17).load("VGG16/VGG16Parameter15")
    layer(19).load("VGG16/VGG16Parameter17")
    layer(21).load("VGG16/VGG16Parameter19")
    layer(24).load("VGG16/VGG16Parameter21")
    layer(26).load("VGG16/VGG16Parameter23")
    layer(28).load("VGG16/VGG16Parameter25")
    //Affine
    layer(31).load("VGG16/VGG16AffineParam1")
    layer(33).load("VGG16/VGG16AffineParam2")
    layer(35).load("VGG16/VGG16AffineParam3")


    //----------data------------
    var data = encodeVGG16((Image.read("VGG16test1.jpg")).map(_.map(_.map(_.toFloat))))

    //--------learning-----------
    for(i<-0 until layer.size){
      if(i==0){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter2")
      }else if(i==2){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter4")
      }else if(i==5){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter6")
      }else if(i==7){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter8")
      }else if(1==10){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter10")
      }else if(i==12){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter12")
      }else if(i==14){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter14")
      }else if(i==17){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter16")
      }else if(i==19){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter18")
      }else if(i==21){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter20")
      }else if(i==24){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter22")
      }else if(i==26){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter24")
      }else if(i==28){
        val Z = layer(i).forward(data)
        data = layer(i).add_b(Z , "VGG16/VGG16Parameter26")
      }else{
        data = layer(i).forward(data)
      }
    }//for



  }//main

}//object
