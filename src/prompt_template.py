import os


example_instructions = {
    "translate_up": {
        "example_1": "高一些",
        "example_2": "画面有点低了",
        "example_3": "可以向上调整一点吗？",
        "example_4": "呃，再往上抬一点点"
    },
    "translate_down": {
        "example_1": "低一些",
        "example_2": "太高了，往下一点",
        "example_3": "麻烦帮我往下调一下好吗？",
        "example_4": "嗯……让它再往下走走"
    },
    "translate_left": {
        "example_1": "向左一点",
        "example_2": "偏右了，往左挪挪",
        "example_3": "请向左边移动一些",
        "example_4": "哎呀，再往左边靠一点"
    },
    "translate_right": {
        "example_1": "向右一点",
        "example_2": "太靠左了，过来点",
        "example_3": "请将画面整体往右挪",
        "example_4": "嗯……可以往右边再来一点"
    },
    "rotate_up": {
        "example_1": "向上转一些",
        "example_2": "手机有点往下斜了",
        "example_3": "请帮我把手机往上转动一点",
        "example_4": "那个，能不能稍微往上转转？"
    },
    "rotate_down": {
        "example_1": "向下转一点",
        "example_2": "画面有点太正了，往下倾斜一下",
        "example_3": "可以向下转一点吗？",
        "example_4": "嗯……手机稍微往下转一下"
    },
    "rotate_left": {
        "example_1": "向左转一点",
        "example_2": "向右斜了，转到左边来一些",
        "example_3": "请往左边旋转一下",
        "example_4": "再往左转一点，再来一点"
    },
    "rotate_right": {
        "example_1": "向右转一点",
        "example_2": "太靠左了，往右边转转",
        "example_3": "麻烦向右转动一些",
        "example_4": "嗯……能不能往右边再转一点"
    },
    "move_forward": {
        "example_1": "近一些",
        "example_2": "太远了，看不清，可以靠近一些吗？",
        "example_3": "请向前移动",
        "example_4": "呃，那个，让它往前边再走走"
    },
    "move_backward": {
        "example_1": "远一些",
        "example_2": "太近了，往后退一点",
        "example_3": "可以向后退一点吗？",
        "example_4": "嗯……离远一点比较好"
    }
}


def load_prompt_template(file_name):
    with open(os.path.join("./prompt/system", file_name), "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    with open(os.path.join("./prompt/user", file_name), "r", encoding="utf-8") as f:
        user_prompt = f.read()
    
    return system_prompt, user_prompt


system_instruction, user_instruction =load_prompt_template('instruction.txt')
system_oos, user_oos = load_prompt_template('oos.txt')
system_speech, user_speech = load_prompt_template('speech_aug.txt')
system_inv, user_inv = load_prompt_template('inv_aug.txt')
system_compound, user_compound = load_prompt_template('compound_aug.txt')
system_adv, user_adv = load_prompt_template('adv_aug.txt')
system_imp, user_imp = load_prompt_template('imp_aug.txt')

system_structured, user_structured = load_prompt_template('structured.txt')


system_gen, user_gen = load_prompt_template('model.txt')
system_annotation, user_annotation = load_prompt_template('model_annotation.txt')