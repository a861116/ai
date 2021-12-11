<template>
  <div class="home">
    <div id="container">
      <div
        style="
          position: absolute;
          top: 0;
          left: 50%;
          font-size: 18px;
          transform: translateX(-50%);
        "
      >
        西瓜的位置：{{ watermelonPostitionx }},{{ watermelonPostitionz }}
      </div>
      <div
        style="
          position: absolute;
          top: 30px;
          left: 50%;
          font-size: 18px;
          transform: translateX(-50%);
        "
      >
        你的位置：{{ personx }},{{ personz }}
      </div>
      <div
        style="
          position: absolute;
          top: 60px;
          left: 50%;
          font-size: 18px;
          transform: translateX(-50%);
        "
      >
        已找到了:{{ level }}个西瓜
      </div>
    </div>
    <div>
      <div id="buttons">
        <button @click="captureSample(0)" class="top">方向：上</button>
        <button @click="captureSample(1)" class="bottom">方向：下</button>
        <button @click="captureSample(2)" class="left">方向：左</button>
        <button @click="captureSample(3)" class="right">方向：右</button>
        <button @click="captureSample(4)" class="middle">方向：空</button>
        <button @click="trainModel()" class="middle1">训练</button>
      </div>
      <div>
        <div><h3>方向上 训练图片数量:{{ upCount }}</h3></div>
        <div><h3>方向下 训练图片数量:{{ downCount }}</h3></div>
        <div><h3>方向左 训练图片数量:{{ leftCount }}</h3></div>
        <div><h3>方向右 训练图片数量:{{ rightCount }}</h3></div>
        <div><h3>方向空 训练图片数量:{{ emptyCount }}</h3></div>
        <div style="font-size:20px;font-weight:600">{{isTranning?'正在训练中···':'等待训练'}}</div>
      </div>
      <video
        autoplay
        playsinline
        muted
        id="webcam"
        width="224"
        height="224"
      ></video>
    </div>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";
import * as Three from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader";
export default {
  name: "ThreeTest",
  data() {
    return {
      camera: null,
      scene: null,
      renderer: null,
      mesh: null,
      controls: "",
      intersections: null,
      objects: [],
      clock: "",
      moveForward: false,
      moveLeft: false,
      moveBackward: false,
      moveRightInit: false,
      direction: new Three.Vector3(),
      velocity: new Three.Vector3(),
      prevTime: performance.now(),
      mixer: null,
      AnimationAction: null,
      isWalk: false,
      timer: null,
      actions: [],
      isRun: false,
      watermelonPostitionx: 0,
      watermelonPostitionz: 0,
      isFound: false,
      personx: 0,
      personz: 0,
      level: 0,
      mobilenet:
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json",
      model: null,
      hasTrained: false,
      trainingData: [],
      webcam: null,
      labels: ["上", "下", "左", "右", "空"],
      upCount: 0,
      downCount: 0,
      leftCount: 0,
      rightCount: 0,
      emptyCount: 0,
      isTranning:false
    };
  },
  methods: {
    personstop() {
      //停止
      if (!this.isRun) {
      } else {
        this.actions[0].play();
        this.actions[1].stop();
        this.isRun = false;
      }
    },
    personMove(category) {
      //判断方向
      let that = this;
      that.checkFound();
      that.personx = that.mesh.position.x;
      that.personz = that.mesh.position.z;
      if (that.isRun) {
      } else {
        that.actions[1].play();
        that.actions[0].stop();
        that.isRun = true;
      }
      switch (category) {
        case 0: //上
          that.mesh.position.z += 4;
          that.camera.position.z = that.mesh.position.z - 80;
          if (!that.mesh.rotation.y == 0) {
            that.mesh.rotation.y = 0;
          }
          break;

        case 2: //左
          that.mesh.position.x += 4;
          that.camera.position.x = that.mesh.position.x;
          if (that.mesh.rotation.y == -300) {
          } else {
            that.mesh.rotation.y = -300;
          }
          break;

        case 1: // 下
          that.mesh.position.z -= 4;
          that.camera.position.z = that.mesh.position.z - 80;
          if (that.mesh.rotation.y == -600) {
          } else {
            that.mesh.rotation.y = -600;
          }

          break;

        case 3: // 右
          that.mesh.position.x -= 4;
          that.camera.position.x = that.mesh.position.x;
          if (that.mesh.rotation.y == 300) {
          } else {
            that.mesh.rotation.y = 300;
          }
          break;
      }
    },

    async predictImage() {
      // 识别部分
      if (!this.hasTrained) {
        return;
      }
      const img = await this.getWebcamImage();
      let result = tf.tidy(() => {
        const input = img.reshape([1, 224, 224, 3]);
        return this.model.predict(input);
      });
      img.dispose();
      let prediction = await result.data();
      result.dispose();
      let id = prediction.indexOf(Math.max(...prediction));
      if (id == 0 || id == 1 || id == 2 || id == 3) {
        console.log(this.labels[id]);
        this.personMove(id);
      } else {
        this.personstop();
      }
    },
    createTransferModel(model) {
      //创建模型
      const bottleneck = model.getLayer("dropout");
      const baseModel = tf.model({
        inputs: model.inputs,
        outputs: bottleneck.output,
      });

      for (const layer of baseModel.layers) {
        layer.trainable = false;
      }

      const newHead = tf.sequential();
      newHead.add(
        tf.layers.flatten({
          inputShape: baseModel.outputs[0].shape.slice(1),
        })
      );
      newHead.add(tf.layers.dense({ units: 100, activation: "relu" }));
      newHead.add(tf.layers.dense({ units: 100, activation: "relu" }));
      newHead.add(tf.layers.dense({ units: 10, activation: "relu" }));
      newHead.add(
        tf.layers.dense({
          units: this.labels.length,
          kernelInitializer: "varianceScaling",
          useBias: false,
          activation: "softmax",
        })
      );

      const newOutput = newHead.apply(baseModel.outputs[0]);
      const newModel = tf.model({
        inputs: baseModel.inputs,
        outputs: newOutput,
      });
      return newModel;
    },
    async trainModel() {
      this.hasTrained = false;
this.isTranning=true
      // 训练模型
      const imageSamples = [];
      const targetSamples = [];
      this.trainingData.forEach((sample) => {
        imageSamples.push(sample.image);
        let cat = [];
        for (let c = 0; c < this.labels.length; c++) {
          cat.push(c === sample.category ? 1 : 0);
        }
        targetSamples.push(tf.tensor1d(cat));
      });
      const xs = tf.stack(imageSamples);
      const ys = tf.stack(targetSamples);

      this.model.compile({
        loss: "meanSquaredError",
        optimizer: "adam",
        metrics: ["acc"],
      });

      await this.model.fit(xs, ys, {
        epochs: 30,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log("Epoch #", epoch, logs);
          },
        },
      });
      this.hasTrained = true;
      this.isTranning=false
    },
    async getWebcamImage() {
      //捕捉画面
      const img = (await this.webcam.capture()).toFloat();
      const normalized = img.div(127).sub(1);
      return normalized;
    },
    async captureSample(category) {
      //捕捉画面
      if (category == 0) {
        this.upCount++;
      } else if (category == 1) {
        this.downCount++;
      } else if (category == 2) {
        this.leftCount++;
      } else if (category == 3) {
        this.rightCount++;
      } else if (category == 4) {
        this.emptyCount++;
      }
      this.trainingData.push({
        image: await this.getWebcamImage(),
        category: category,
      });
    },
    async setupWebcam() {
      //初始化摄像头
      return new Promise((resolve, reject) => {
        const webcamElement = document.getElementById("webcam");
        const navigatorAny = navigator;
        navigator.getUserMedia =
          navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia ||
          navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
          navigator.getUserMedia(
            { video: true },
            (stream) => {
              webcamElement.srcObject = stream;
              webcamElement.addEventListener("loadeddata", resolve, false);
            },
            (error) => reject()
          );
        } else {
          reject();
        }
      });
    },

    init: async function () {
      this.initScene(); //场景对象
      this.initCamera(); //相机
      this.initWebGLRenderer(); //渲染器
      this.render(); //渲染
      this.createControls(); //控件对象
      this.initLight(); //灯光
      this.initModel(); //人
      this.Watermelon(); //西瓜
      this.initControls(); //相机视角
      this.initPlane(); //地板

      //识别部分
      this.model = await tf.loadLayersModel(this.mobilenet);
      this.model = this.createTransferModel(this.model);
      await this.setupWebcam();
      this.webcam = await tf.data.webcam(document.getElementById("webcam"));
      setInterval(this.predictImage, 50);
    },

    initControls() {
      let that = this;
      that.controls = new PointerLockControls(this.camera, document.body);
      let container = document.getElementById("container");
      container.addEventListener("click", function () {
        that.controls.lock();
      });
      this.scene.add(that.controls.getObject());
    },

    //创建场景对象Scene
    initScene() {
      this.scene = new Three.Scene();
      this.scene.background = new Three.Color(0xa0a0a0);
      this.scene.fog = new Three.Fog(0xa0a0a0, 1000, 11000);
    },
    //相机
    initCamera() {
      let container = document.getElementById("container");
      this.camera = new Three.PerspectiveCamera(
        50,
        container.clientWidth / container.clientHeight,
        0.01,
        20000
      );
      this.camera.position.set(0, 40, -80); //设置相机位置
      this.camera.lookAt(new Three.Vector3(0, 1, 0)); //设置相机方向(指向的场景对象)
    },
    Watermelon() {
      //西瓜
      let sphere = new Three.TextureLoader().load("/2.jpg");
      sphere.wrapS = sphere.wrapT = Three.RepeatWrapping;
      sphere.repeat.set(1.2, 1.2);
      sphere.anisotropy = 36;
      let sphereMaterial = new Three.MeshLambertMaterial({
        map: sphere,
      });

      const geometry = new Three.SphereGeometry(10, 10, 10);
      let plusOrMinus = Math.random() < 0.5 ? -1 : 1;
      let plusOrMinus1 = Math.random() < 0.5 ? -1 : 1;

      const cube = new Three.Mesh(geometry, sphereMaterial);
      cube.name = "西瓜";
      this.watermelonPostitionx =
        (Math.random() * 600).toFixed(0) * plusOrMinus;
      this.watermelonPostitionz =
        (Math.random() * 600).toFixed(0) * plusOrMinus1;
      cube.position.set(
        this.watermelonPostitionx,
        10,
        this.watermelonPostitionz
      );
      cube.receiveShadow = true;
      cube.castShadow = true;
      this.scene.add(cube);
    },
    initModel() {
      let that = this;
      let fbxLoader = new FBXLoader();
      fbxLoader.load("/2.FBX", function (mesh) {
        mesh.scale.set(0.1, 0.1, 0.1);
        that.mesh = mesh;
        that.mesh.traverse(function (child) {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        that.scene.add(that.mesh);

        // obj作为参数创建一个混合器，解析播放obj及其子对象包含的动画数据
        that.mixer = new Three.AnimationMixer(that.mesh);
        // 查看动画数据

        // mesh.animations[0]：获得剪辑对象clip

        that.actions = []; //所有的动画数组
        for (let i = 0; i < that.mesh.animations.length; i++) {
          that.actions[i] = that.mixer.clipAction(that.mesh.animations[i]);
        }
        that.actions[0].play();
        // AnimationAction.loop = THREE.LoopOnce; //不循环播放
        // AnimationAction.clampWhenFinished=true;//暂停在最后一帧播放的状态
      });
    },
    //地板
    initPlane() {
      let groundTexture = new Three.TextureLoader().load("/grass.jpg");
      groundTexture.wrapS = groundTexture.wrapT = Three.RepeatWrapping;
      groundTexture.repeat.set(250, 250);
      groundTexture.anisotropy = 36;
      let groundMaterial = new Three.MeshLambertMaterial({
        map: groundTexture,
      });
      let mesh = new Three.Mesh(
        new Three.PlaneBufferGeometry(20000, 20000),
        groundMaterial
      );
      mesh.rotation.x = -Math.PI / 2;
      mesh.receiveShadow = true;
      this.scene.add(mesh);
    },
    //创建渲染器对象
    initWebGLRenderer() {
      this.renderer = new Three.WebGLRenderer({
        antialias: true,
      });
      this.renderer.setSize(container.clientWidth, container.clientHeight); //设置渲染区域尺寸
      this.renderer.setClearColor(0xb9d3ff, 1); //设置背景颜色
      this.renderer.shadowMap.enabled = true;
      container.appendChild(this.renderer.domElement); //body元素中插入canvas对象
    },
    initLight() {
      const hemiLight = new Three.AmbientLight(0x444444);

      this.scene.add(hemiLight);

      const dirLight = new Three.DirectionalLight(0xffffff);
      dirLight.position.set(2000, 2000, 2000);
      dirLight.castShadow = true;
      dirLight.shadow.mapSize.width = dirLight.shadow.mapSize.height = 1024;
      dirLight.shadow.camera.top = 500;
      dirLight.shadow.camera.bottom = -500;
      dirLight.shadow.camera.left = -500;
      dirLight.shadow.camera.right = 500;
      dirLight.shadow.camera.near = 200;
      dirLight.shadow.camera.far = 200000;

      this.scene.add(dirLight);
    },
    checkFound() {
      //计算是否找到西瓜
      let that = this;
      let distance = Math.pow(
        Math.pow(this.mesh.position.x - this.watermelonPostitionx, 2) +
          Math.pow(this.mesh.position.z - this.watermelonPostitionz, 2),
        0.5
      );
      if (distance < 30 && this.isFound == false) {
        this.isFound = true;
        this.level++;
        this.scene.children.forEach((element) => {
          if (element.name == "西瓜") {
            that.scene.remove(element);
          }
          this.isFound = false;
        });

        that.Watermelon();
      }
    },
    render: function () {
      let that = this;
      //this.mesh.rotation.x+=0.01
      requestAnimationFrame(that.render); //请求再次执行渲染函数render
      that.renderer.render(that.scene, that.camera); //执行渲染操作

      //从这开始下面**就是控制移动的代码
      let time = performance.now();
      let delta = (time - that.prevTime) / 1000;
      if (that.mixer !== null) {
        //clock.getDelta()方法获得两帧的时间间隔
        // 更新混合器相关的时间
        that.mixer.update(delta);
      }
      that.velocity.x -= that.velocity.x * 10.0 * delta;
      that.velocity.z -= that.velocity.z * 10.0 * delta;

      that.direction.z = Number(that.moveForward) - Number(that.moveBackward);
      that.direction.x = Number(that.moveRightInit) - Number(that.moveLeft);

      that.velocity.z -= that.direction.z * 400.0 * delta;
      that.velocity.x -= that.direction.x * 400.0 * delta;

      //下面两个&&是判断是否有moveRight。如果有就执行。不然没加载完毕，就循环执行会报错
      that.controls && that.controls.moveRight(-that.velocity.x * delta);
      that.controls && that.controls.moveForward(-that.velocity.z * delta);
      that.prevTime = time;
    },
    // 创建控件对象
    createControls() {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    },
  },
  mounted() {
    this.init();
    this.render();
  },
};
</script>
<style scoped>
#container {
  height: 100%;
  width: 80%;
  position: relative;
}
.home {
  display: flex;
  height: 100vh;
  width: 100vw;
}
#buttons{
  position:relative;
	width: 320px;
	height: 320px;
}
button{
  padding: 1rem;
  border: 1px white solid;
display: block;
background-color: skyblue;
color: white;
border-radius: 20px;
cursor: pointer;
}
.left {
  
	position: absolute;
	top: 50%;
	left: 2%; 
  transform: translateY(-50%);
}
/*bottom*/
.bottom{
  
	position: absolute;
	bottom:2%;
	left: 50%;
  transform: translateX(-50%);
}
/*right*/
.right{
  
	position: absolute;
	top: 50%;
	right: 2%;
  transform: translateY(-50%);
}
/*top*/
.top{
  
	position: absolute;
	top: 2%;
	left: 50%; 
  transform: translateX(-50%);
}
.middle{
left: 50%; 
  transform: translateX(-50%);
  top: 35%;
  position: absolute;
}
.middle1{
left: 50%; 
  transform: translateX(-50%);
  bottom:  35%;
  position: absolute;
}

</style>
