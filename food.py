import streamlit as st
import random
from PIL import Image
import os
import time

# 设置页面配置
st.set_page_config(
    page_title="随机美食推荐器",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="expanded"
)


# 获取图片绝对路径
def get_image_path(image_name):
    # 根据你的实际位置调整路径
    return r"C:\Users\Lenovo\Desktop\images" + "\\" + image_name


# 加载CSS样式
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"加载样式表错误: {e}")
        st.markdown("""
        <style>
        /* 基础样式作为后备 */
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF2B2B;
            transform: scale(1.05);
        }
        </style>
        """, unsafe_allow_html=True)


# 应用CSS样式
local_css(r"C:\Users\Lenovo\Desktop\style.css")

# 应用标题和简介
st.title("🍜 随机美食推荐器")
st.markdown("""
探索世界各地的美食！选择一个菜系，我们将为您随机推荐一道经典菜肴，并附上详细介绍和烹饪小贴士。
""")

# 美食数据库
food_database = {
    "中餐": [
        {
            "name": "宫保鸡丁",
            "image": "kung_pao_chicken.jpg",
            "description": "宫保鸡丁是一道经典的川菜，以鸡肉、花生米、辣椒和花椒为主要原料，口味麻辣鲜香，略带甜酸。",
            "tips": "关键技巧：鸡丁先用蛋清和淀粉腌制，保持肉质嫩滑；最后加入花生米快速翻炒，保持脆度。"
        },
        {
            "name": "麻婆豆腐",
            "image": "mapo_tofu.jpg",
            "description": "麻婆豆腐是四川传统名菜，主要原料为豆腐、牛肉末、辣椒和花椒，具有麻、辣、烫、香、酥、嫩、鲜、活的特点。",
            "tips": "使用嫩豆腐效果最佳；最后撒上花椒粉和葱花，风味更佳。"
        },
        {
            "name": "北京烤鸭",
            "image": "peking_duck.jpg",
            "description": "北京烤鸭享誉世界，以其皮脆肉嫩、色泽红艳而闻名。传统吃法是用薄饼包裹鸭肉、黄瓜条和甜面酱。",
            "tips": "鸭子需经过吹气、烫皮、挂糖等多道工序；烤制时火候控制是关键。"
        }
    ],
    "意大利餐": [
        {
            "name": "玛格丽特披萨",
            "image": "margherita_pizza.jpg",
            "description": "经典意大利披萨，以番茄酱、新鲜马苏里拉奶酪和罗勒叶为配料，代表意大利国旗的红、白、绿三色。",
            "tips": "面团需充分发酵；使用石窑烤炉能获得最佳口感；新鲜罗勒叶在出炉后添加。"
        },
        {
            "name": "意大利肉酱面",
            "image": "spaghetti_bolognese.jpg",
            "description": "博洛尼亚传统美食，由牛肉末、番茄酱、洋葱、胡萝卜和芹菜慢炖数小时制成，配以意大利面。",
            "tips": "肉酱需小火慢炖至少2小时；使用宽面条如tagliatelle更传统。"
        },
        {
            "name": "提拉米苏",
            "image": "tiramisu.jpg",
            "description": "意大利经典甜点，由浸泡过咖啡和朗姆酒的手指饼干、马斯卡彭奶酪和可可粉层层叠加而成。",
            "tips": "使用新鲜鸡蛋；奶酪需室温软化；冷藏至少4小时后再享用。"
        }
    ],
    "日本料理": [
        {
            "name": "寿司拼盘",
            "image": "sushi_platter.jpg",
            "description": "日本代表性美食，由醋饭和各种海鲜组成。常见的有金枪鱼、三文鱼、虾和鳗鱼等。",
            "tips": "米饭需用专用寿司醋调味；新鲜海鲜是关键；芥末应抹在鱼生而非酱油中。"
        },
        {
            "name": "拉面",
            "image": "ramen.jpg",
            "description": "日式汤面，由浓郁骨汤、面条和各种配料（如叉烧肉、溏心蛋、海苔、竹笋）组成。",
            "tips": "汤底需熬制8小时以上；面条煮至al dente状态；溏心蛋煮6分钟最佳。"
        },
        {
            "name": "天妇罗",
            "image": "tempura.jpg",
            "description": "日式炸物，将海鲜和蔬菜裹上特制面糊后油炸而成，以酥脆轻盈的口感为特点。",
            "tips": "面糊需冷藏且不过度搅拌；油温控制在170-180°C；立即享用口感最佳。"
        }
    ],
    "墨西哥餐": [
        {
            "name": "塔可",
            "image": "tacos.jpg",
            "description": "墨西哥传统食物，由玉米或小麦薄饼包裹各种馅料，如牛肉、鸡肉、蔬菜、莎莎酱和鳄梨酱。",
            "tips": "玉米饼需加热软化；新鲜制作的莎莎酱风味更佳；添加青柠汁提鲜。"
        },
        {
            "name": "墨西哥卷饼",
            "image": "burrito.jpg",
            "description": "大尺寸面粉薄饼包裹米饭、豆类、肉类、奶酪和莎莎酱等丰富馅料，营养均衡且饱腹感强。",
            "tips": "馅料需沥干水分；卷饼时需紧密包裹；可烘烤使表面酥脆。"
        },
        {
            "name": "鳄梨酱",
            "image": "guacamole.jpg",
            "description": "以鳄梨为主料，加入番茄、洋葱、香菜和青柠汁制成的经典墨西哥蘸酱，清新健康。",
            "tips": "使用成熟但不过软的鳄梨；立即加入青柠汁防止氧化；保留果核延缓变色。"
        }
    ],
    "印度菜": [
        {
            "name": "咖喱鸡",
            "image": "chicken_curry.jpg",
            "description": "印度最受欢迎的菜肴之一，鸡肉在由多种香料熬制的浓郁咖喱酱汁中慢炖而成，配以米饭或烤饼。",
            "tips": "使用完整香料干炒释放香气；加入酸奶使肉质更嫩；最后加入奶油增加浓郁口感。"
        },
        {
            "name": "印度烤饼",
            "image": "naan.jpg",
            "description": "传统印度发酵面饼，在泥炉中烤制而成，外酥内软，常搭配咖喱食用。",
            "tips": "面团需充分发酵；烤制前涂抹黄油或蒜蓉；趁热食用口感最佳。"
        },
        {
            "name": "印度香饭",
            "image": "biryani.jpg",
            "description": "印度王室菜肴，由长粒香米、肉类、蔬菜和多种香料层层叠加蒸制而成，香气浓郁。",
            "tips": "米饭预先半熟；使用藏红花水染色；密封容器蒸制保持香气。"
        }
    ]
}


# 创建占位图片
def create_placeholder_image(text, width=400, height=300):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (width, height), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    try:
        # 尝试加载中文字体
        font = ImageFont.truetype("simhei.ttf", 30)
    except:
        try:
            # 尝试其他常见字体
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            # 最后使用默认字体
            font = ImageFont.load_default()

    # 计算文本位置
    text_width = font.getlength(text)
    text_height = 30  # 近似值
    draw.text(((width - text_width) // 2, (height - text_height) // 2),
              text, font=font, fill=(255, 255, 255))
    return img


# 加载食物图片
def load_food_image(image_name):
    try:
        image_path = get_image_path(image_name)
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.warning(f"图片未找到: {image_path}")
            return create_placeholder_image(image_name.split('.')[0].replace('_', ' ').title())
    except Exception as e:
        st.error(f"加载图片错误: {e}")
        return create_placeholder_image("图片加载失败")


# 侧边栏
with st.sidebar:
    st.header("🍽️ 选择你的美食之旅")
    cuisine = st.selectbox("选择菜系", list(food_database.keys()), index=0)

    st.markdown("---")
    st.markdown("### 烹饪偏好")
    spice_level = st.slider("辣度偏好", 0, 5, 2)
    cooking_time = st.selectbox("烹饪时间", ["15分钟内", "30分钟", "1小时", "2小时以上"])

    st.markdown("---")
    st.markdown("### 关于本应用")
    st.markdown("""
    这个随机美食推荐器旨在帮助您：
    - 发现世界各地美食
    - 尝试新的烹饪体验
    - 获取专业烹饪技巧
    - 拓展您的美食视野
    """)

# 推荐按钮
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎲 随机推荐美食", use_container_width=True):
        # 获取随机美食
        foods = food_database[cuisine]
        selected_food = random.choice(foods)

        # 显示加载动画
        with st.spinner(f'正在为您挑选完美的{cuisine}美食...'):
            time.sleep(1.5)

        # 显示结果
        st.subheader(f"今日推荐：{selected_food['name']}")

        # 显示美食图片
        food_image = load_food_image(selected_food["image"])
        st.image(food_image, caption=selected_food["name"], use_column_width=True)

        # 显示美食信息
        st.markdown(f"**菜系**: {cuisine}")
        st.markdown(f"**简介**: {selected_food['description']}")
        st.markdown(f"**👨‍🍳 烹饪小贴士**: {selected_food['tips']}")

        # 添加一些装饰元素
        st.balloons()
        st.success("发现美食成功！祝您烹饪愉快！")

        # 营养信息（模拟）
        with st.expander("📊 营养信息（每份）"):
            st.markdown("""
            | 营养成分 | 含量 |
            |----------|------|
            | 卡路里   | 350-550 kcal |
            | 蛋白质   | 20-30g |
            | 碳水化合物 | 40-60g |
            | 脂肪     | 15-25g |
            """)

        # 用户反馈
        st.markdown("---")
        st.subheader("您的反馈")
        rating = st.slider("您喜欢这个推荐吗？", 1, 5, 3)
        if st.button("提交反馈"):
            st.success(f"感谢您的评分：{rating}颗星！我们将根据您的反馈改进推荐。")
    else:
        st.info("👆 点击上方按钮获取随机美食推荐！")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 10px;">
    <p>🍴 用代码烹饪美食 | 随机美食推荐器 © 2023</p>
    <p>探索世界美食，开启味蕾之旅</p>
</div>
""", unsafe_allow_html=True)