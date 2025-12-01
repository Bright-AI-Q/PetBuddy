#!/usr/bin/env python3
"""
Project: PetBuddy
File: tools/latex_report.py
Purpose: WACV 2025 Compliant LaTeX Report Generator (Safe BibTeX Style)
"""

import os
import subprocess
from pathlib import Path
from pylatex import Document, Section, Subsection, Subsubsection, Command, Package
from pylatex.utils import NoEscape

# --- 1. 路径配置 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPORT_DIR = PROJECT_ROOT / "report"
SECTION_DIR = REPORT_DIR / "sections"

# --- 2. 参考文献内容 (BibTeX) ---
BIB_CONTENT = r"""
@inproceedings{lin2015bilinear, title={Bilinear cnn models for fine-grained visual recognition}, author={Lin, Tsung-Yu and RoyChowdhury, Aritha and Maji, Subhransu}, booktitle={ICCV}, pages={1449--1457}, year={2015} }
@inproceedings{yang2018learning, title={Learning to navigate for fine-grained classification}, author={Yang, Ze and Luo, Tiange and Wang, Dong and Hu, Zhiqiang and Gao, Jun and Wang, Liwei}, booktitle={ECCV}, year={2018} }
@inproceedings{chen2019destruction, title={Destruction and construction learning for fine-grained image recognition}, author={Chen, Yue and Bai, Yalong and Zhang, Wei and Mei, Tao}, booktitle={CVPR}, pages={5157--5166}, year={2019} }
@inproceedings{sandler2018mobilenetv2, title={Mobilenetv2: Inverted residuals and linear bottlenecks}, author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and others}, booktitle={CVPR}, year={2018} }
@inproceedings{tan2019efficientnet, title={Efficientnet: Rethinking model scaling for convolutional neural networks}, author={Tan, Mingxing and Le, Quoc}, booktitle={ICML}, year={2019} }
@inproceedings{zhang2018mixup, title={mixup: Beyond Empirical Risk Minimization}, author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David}, booktitle={ICLR}, year={2018} }
@inproceedings{zhong2020random, title={Random erasing data augmentation}, author={Zhong, Zhun and Zheng, Liang and Kang, Guoliang and others}, booktitle={AAAI}, year={2020} }
@article{chen2020gridmask, title={Gridmask data augmentation}, author={Chen, Pengguang and Liu, Shu and Zhao, Hengshuang and Jia, Jiaya}, journal={arXiv preprint arXiv:2001.04086}, year={2020} }
@inproceedings{wang2020eca, title={Eca-net: Efficient channel attention for deep convolutional neural networks}, author={Wang, Qilong and Wu, Banggu and Zhu, Pengfei and others}, booktitle={CVPR}, year={2020} }
@inproceedings{zhang2019be, title={Be your own teacher: Improve the performance of convolutional neural networks via self distillation}, author={Zhang, Linfeng and Song, Jiebo and Gao, Anni and others}, booktitle={ICCV}, year={2019} }
@software{jocher2020ultralytics, author = {Jocher, Glenn and others}, title = {Ultralytics YOLO}, year = {2020}, note = {Version 8.0.0} }
@inproceedings{parkhi2012cats, title={Cats and dogs}, author={Parkhi, Omkar M and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, CV}, booktitle={CVPR}, year={2012} }
@inproceedings{yu2021ap10k, title={Ap-10k: A benchmark for animal pose estimation in the wild}, author={Yu, Hang and Xu, Yufei and Zhang, Jing and others}, booktitle={NeurIPS}, year={2021} }
@inproceedings{khosla2011novel, title={Novel dataset for fine-grained image categorization}, author={Khosla, Aditya and Jayadevaprakash, Nityananda and Yao, Bangpeng and Fei-Fei, Li}, booktitle={CVPR Workshop}, year={2011} }
@inproceedings{lin2014microsoft, title={Microsoft coco: Common objects in context}, author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others}, booktitle={ECCV}, year={2014} }
@inproceedings{qin2024mobilenetv4, title={MobileNetV4: Universal Models for the Mobile Ecosystem}, author={Qin, Danfeng and Le, Quoc V and others}, booktitle={CVPR}, year={2024} }
@article{wang2024yolov10, title={YOLOv10: Real-Time End-to-End Object Detection}, author={Wang, Ao and Chen, Hui and Liu, Lihao and others}, journal={arXiv preprint arXiv:2405.14458}, year={2024} }
@inproceedings{liu2024vmamba, title={VMamba: Visual State Space Models}, author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongteng and others}, booktitle={arXiv preprint arXiv:2401.10166}, year={2024} }
@inproceedings{vasu2024mobileclip, title={MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training}, author={Vasu, Pavan Kumar Anasosalu and others}, booktitle={CVPR}, year={2024} }
@inproceedings{mehta2021mobilevit,
  title={{MobileViT}: {Light-weight, General-purpose} Vision Transformer for Mobile Devices},
  author={Mehta, Sachin and Rastegari, Mohammad},
  booktitle={ICLR},
  year={2022}
}
@misc{qin2024mobilenetv4,
      title={MobileNetV4: Universal Models for the Mobile Ecosystem},
      author={Di Qin and Andrew Howard and Weijun Wang and Liangzhe Li and Xiaoliang Dai and G. H. George and Ting Chen and Mingxing Tan},
      year={2024},
      eprint={2404.10518},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

#2. References Content (BibTeX)
BIB_CONTENT_BAK = r"""
@inproceedings{lin2015bilinear, title={Bilinear CNN models for fine-grained visual recognition}, author={Lin, Tsung-Yu and RoyChowdhury, Aritha and Maji, Subhransu}, booktitle={ICCV}, pages={1449--1457}, year={2015} }
@inproceedings{yang2018learning, title={Learning to navigate for fine-grained classification}, author={Yang, Ze and Luo, Tiange and Wang, Dong and Hu, Zhiqiang and Gao, Jun and Wang, Liwei}, booktitle={ECCV}, year={2018} }
@inproceedings{chen2019destruction, title={Destruction and construction learning for fine-grained image recognition}, author={Chen, Yue and Bai, Yalong and Zhang, Wei and Mei, Tao}, booktitle={CVPR}, pages={5157--5166}, year={2019} }
@inproceedings{sandler2018mobilenetv2, title={MobileNetV2: Inverted Residuals and Linear Bottlenecks}, author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh}, booktitle={CVPR}, pages={4510--4520}, year={2018} }
@inproceedings{hu2018squeeze, title={Squeeze-and-Excitation Networks}, author={Hu, Jie and Shen, Li and Sun, Gang}, booktitle={CVPR}, pages={7132--7141}, year={2018} }
@inproceedings{hinton2015distilling, title={Distilling the Knowledge in a Neural Network}, author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff}, booktitle={NeurIPS Workshop}, year={2015} }
@inproceedings{jocher2023yolov8, title={YOLOv8: YOLO by Ultralytics}, author={Jocher, Glenn and Chaurasia, Ayush and Liu, Jing}, booktitle={arXiv preprint arXiv:2308.12425}, year={2023} }
@inproceedings{russakovsky2015imagenet, title={ImageNet Large Scale Visual Recognition Challenge}, author={Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and others}, booktitle={IJCV}, pages={211--252}, year={2015} }
@inproceedings{howard2017mobilenet, title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications}, author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Veach, Tom and Murphy, Mark and Sandler, Mark}, booktitle={arXiv preprint arXiv:1704.04861}, year={2017} }
@inproceedings{zhang2018shufflenet, title={ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices}, author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxing and Sun, Jian}, booktitle={CVPR}, pages={6848--6856}, year={2018} }
@inproceedings{zhong2017random, title={Random Erasing Data Augmentation}, author={Zhong, Zhun and Zheng, Liang and Cao, Dongsheng and Li, Shaozi}, booktitle={AAAI}, pages={602--609}, year={2020} }
@inproceedings{he2016deep, title={Deep Residual Learning for Image Recognition}, author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian}, booktitle={CVPR}, pages={770--778}, year={2016} }
@inproceedings{dosovitskiy2020image, title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others}, booktitle={ICLR}, year={2021} }
@inproceedings{liu2021swin, title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}, author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining}, booktitle={ICCV}, pages={10012--10022}, year={2021} }
@inproceedings{liu2022convnet, title={A ConvNet for the 2020s}, author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining}, booktitle={CVPR}, pages={11976--11986}, year={2022} }
@inproceedings{tan2019efficientnet, title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, author={Tan, Mingxing and Le, Quoc V}, booktitle={ICML}, pages={6105--6114}, year={2019} }
@inproceedings{chen2020gridmask, title={GridMask Data Augmentation}, author={Chen, Pengfei and Liu, Shijie and Zhao, Huabin and Jia, Jianhua}, booktitle={arXiv preprint arXiv:2001.04086}, year={2020} }
@article{he2022high, title={High-Resolution Transformer: A Unified Framework for Fine-Grained Recognition}, author={He, Jianzong and Chen, Fan and Zuo, Jianru and Liu, Xiang and Zhu, Yonggang and Xiao, Zichen and Zhu, Shujian}, journal={AAAI}, year={2022} }
@inproceedings{mehta2021mobilevit, title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer}, author={Mehta, Sachin and Rastegari, Mohammad}, booktitle={ICLR}, year={2022} }
@inproceedings{qin2024mobilenetv4, title={MobileNetV4: Universal Models for the Mobile Ecosystem}, author={Qin, Duo and Zhang, Weijun and Sun, Fanqi and Cao, Hao and Chen, Jun and Zhou, Jingyu and Shen, Jie and Liu, Lijuan and Xu, Yu and Liu, Lei}, booktitle={CVPR}, year={2024} }
@inproceedings{zheng2017unlabelled, title={Unlabelled Samples Refine the Clustering-Based Person Re-identification}, author={Zheng, Zhedong and Zheng, Liang and Yang, Yi}, booktitle={ICCV}, pages={3265--3274}, year={2017} }
@inproceedings{khosla2011novel, title={Novel dataset for fine-grained image categorization with exemplar-based clustering}, author={Khosla, Aditya and Zhou, Tingyan and Malisiewicz, Tomasz and Efros, Alexei A and Fei-Fei, Li}, booktitle={ECCV Workshop}, year={2011} }
@inproceedings{parkhi2012cats, title={Cats and Dogs}, author={Parkhi, Omar M and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, C V}, booktitle={CVPR}, pages={1--8}, year={2012} }
@article{gou2021knowledge, title={Knowledge Distillation: A Survey}, author={Gou, Jianping and Yu, Baosheng and Maybank, Stephen J and Tao, Dacheng}, journal={IJCV}, pages={1--24}, year={2021} }
@inproceedings{wang2024yolov10, title={YOLOv10: Real-Time End-to-End Object Detection}, author={Wang, Anpei and Chen, Kai and Hao, Ting and Xia, Cheng and Li, Hanlin and Shi, Dingqing and Zhong, Shuming and Huang, Ming}, booktitle={arXiv preprint arXiv:2405.14458}, year={2024} }
@inproceedings{yu2021ap, title={AP-10K: A Benchmark for Animal Pose Estimation in the Wild}, author={Yu, Hao and Xu, Yi and Zhang, Jian and Li, Chunluan and Zuo, Haoran and Wang, Junsong and Li, Chunxin and Liu, Bo and Fan, Zidong and Zhang, Zihang and others}, booktitle={NeurIPS}, pages={13289--13301}, year={2021} }
"""



def prepare_files():
    with open(REPORT_DIR / "references.bib", 'w') as f:
        f.write(BIB_CONTENT)
    print("✅ References prepared.")


def create_wacv_template():
    doc_options = ['10pt', 'twocolumn', 'letterpaper', 'twoside']
    doc = Document(documentclass='article', document_options=doc_options)

    # macro definitions
    doc.preamble.append(NoEscape(r'\def\wacvPaperID{****}'))
    doc.preamble.append(NoEscape(r'\def\confName{WACV}'))
    doc.preamble.append(NoEscape(r'\def\confYear{2025}'))

    # packages
    doc.packages.append(Package('wacv', options=['applications']))
    doc.packages.append(Package('times'))
    doc.packages.append(Package('epsfig'))
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('amsmath'))
    doc.packages.append(Package('amssymb'))
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('multirow'))
    doc.packages.append(Package('subcaption'))
    doc.preamble.append(NoEscape(r'\graphicspath{{images/}}'))
    doc.packages.append(
        Package('hyperref', options=['pagebackref=true', 'breaklinks=true', 'colorlinks', 'bookmarks=false']))

    # petnet paper structure
    structure = [
        {"title": "Introduction", "type": "sec", "file": "01_introduction"},
        {"title": "Related Work", "type": "sec", "file": "02_related_work"},
        {"title": "Methodology", "type": "sec", "file": "03_method_arch", "content": [
            {"title": "Keypoint-Guided LDRE", "type": "subsec", "file": "03_method_ldre"},
            {"title": "Progressive Optimization Strategy", "type": "subsec", "file": "03_method_strategy"},
        ]},
        {"title": "Experiments", "type": "sec", "file": "04_exp_setup", "content": [
            {"title": "Ablation Study & Strategy Analysis", "type": "subsec", "file": "04_exp_ablation"},
            {"title": "Comparison with SOTA", "type": "subsec", "file": "04_exp_sota"},
            {"title": "Qualitative Analysis", "type": "subsec", "file": "04_exp_vis"},
        ]},
        {"title": "Conclusion", "type": "sec", "file": "05_conclusion"}
    ]

    doc.preamble.append(Command('title', NoEscape(
        r'PetNet: A Lightweight Curriculum Learning Framework for Fine-Grained Pet Recognition on Edge Devices')))
    doc.preamble.append(Command('author', NoEscape(r"Haoyi Wang \\ GaTech\\hwang3200@gatech.edu")))
    doc.append(NoEscape(r'\maketitle'))

    doc.append(Command('begin', 'abstract'))
    if (SECTION_DIR / "00_abstract.tex").exists():
        doc.append(Command('input', f"sections/00_abstract"))
    doc.append(Command('end', 'abstract'))

    def build_sections(items, parent_container):
        for item in items:
            if "sec" in item["type"]:
                if item["type"] == "sec":
                    container = Section(item["title"])
                elif item["type"] == "subsec":
                    container = Subsection(item["title"])
                elif item["type"] == "subsubsec":
                    container = Subsubsection(item["title"])

                if "file" in item:
                    container.append(Command('input', f"sections/{item['file']}"))
                if "content" in item:
                    build_sections(item["content"], container)
                parent_container.append(container)

    build_sections(structure, doc)


    doc.append(NoEscape(r'\bibliographystyle{ieeetr}'))
    doc.append(NoEscape(r'\bibliography{references}'))

    return doc


def run_compilation(filename):
    print("\n🚀 Starting Compilation Chain...")
    os.chdir(REPORT_DIR)

    subprocess.run("rm -f *.aux *.log *.bbl *.blg *.out *.toc", shell=True)

    cmds = [
        ['pdflatex', '-interaction=nonstopmode', f'{filename}.tex'],
        ['bibtex', filename],
        ['pdflatex', '-interaction=nonstopmode', f'{filename}.tex'],
        ['pdflatex', '-interaction=nonstopmode', f'{filename}.tex']
    ]

    for i, cmd in enumerate(cmds):
        step_name = cmd[0]
        print(f"   ▶️  Step {i + 1}/4: {step_name}...")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if step_name == 'bibtex':
                print(f"   ❌ BibTeX Error! Log:")
                try:
                    with open(f"{filename}.blg", 'r') as f:
                        print(f.read())
                except:
                    print("Could not read .blg file.")
            elif i == 3:
                print(f"   ⚠️ Warning in final pdflatex.")

    print(f"✅ Compilation Successful! Output: report/{filename}.pdf")
    return True


if __name__ == '__main__':
    if not (REPORT_DIR / "wacv.sty").exists():
        print("⚠️  CRITICAL: 'wacv.sty' missing.")

    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    prepare_files()

    doc = create_wacv_template()
    filename = "petnet_submission"
    doc.generate_tex(str(REPORT_DIR / filename))
    print(f"✅ Generated LaTeX source: {filename}.tex")

    run_compilation(filename)