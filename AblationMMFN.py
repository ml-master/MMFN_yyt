import torch


def ablation_experiment():
    # 设置不同的消融实验名称
    experiments = ["full_model", "no_unimodal", "no_crossmodule", "no_transformer", "no_clip", "no_image"]

    for experiment in experiments:
        print(f"Running experiment: {experiment}")
        # 初始化模型
        model = MultiModal()

        # 根据实验名称调整模型结构
        if experiment == "no_unimodal":
            def forward_no_unimodal(input_ids, attention_mask, token_type_ids, image_raw, text, image):
                BERT_feature = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                last_hidden_states = BERT_feature['last_hidden_state']
                text_raw = torch.sum(last_hidden_states, dim=1) / 300
                image_raw = self.swin(image_raw)

                text_m = self.t_projection_net(last_hidden_states)
                image_m = self.i_projection_net(image_raw.last_hidden_state)
                text_att, image_att = self.trans(text_m, image_m)
                correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
                sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
                sim = sim * self.w + self.b
                mweight = sim.unsqueeze(1)
                correlation = correlation * mweight
                final_feature = torch.cat([correlation], 1)
                pre_label = self.classifier_corre(final_feature)
                return pre_label
            model.forward = forward_no_unimodal

        elif experiment == "no_crossmodule":
            def forward_no_crossmodule(input_ids, attention_mask, token_type_ids, image_raw, text, image):
                BERT_feature = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                last_hidden_states = BERT_feature['last_hidden_state']
                text_raw = torch.sum(last_hidden_states, dim=1) / 300
                image_raw = self.swin(image_raw)

                text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image], 1))
                text_m = self.t_projection_net(last_hidden_states)
                image_m = self.i_projection_net(image_raw.last_hidden_state)
                text_att, image_att = self.trans(text_m, image_m)
                sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
                sim = sim * self.w + self.b
                mweight = sim.unsqueeze(1)
                correlation = torch.cat([text_att, image_att], 1) * mweight
                final_feature = torch.cat([text_prime, image_prime], 1)
                pre_label = self.classifier_corre(final_feature)
                return pre_label
            model.forward = forward_no_crossmodule

        elif experiment == "no_transformer":
            def forward_no_transformer(input_ids, attention_mask, token_type_ids, image_raw, text, image):
                BERT_feature = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                last_hidden_states = BERT_feature['last_hidden_state']
                text_raw = torch.sum(last_hidden_states, dim=1) / 300
                image_raw = self.swin(image_raw)

                text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image], 1))
                text_m = self.t_projection_net(last_hidden_states)
                image_m = self.i_projection_net(image_raw.last_hidden_state)
                correlation = self.cross_module(text, image, text_m, image_m)
                sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
                sim = sim * self.w + self.b
                mweight = sim.unsqueeze(1)
                correlation = correlation * mweight
                final_feature = torch.cat([text_prime, image_prime, correlation], 1)
                pre_label = self.classifier_corre(final_feature)
                return pre_label
            model.forward = forward_no_transformer

        elif experiment == "no_clip":
            def forward_no_clip(input_ids, attention_mask, token_type_ids, image_raw, text, image):
                BERT_feature = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                last_hidden_states = BERT_feature['last_hidden_state']
                text_raw = torch.sum(last_hidden_states, dim=1) / 300
                image_raw = self.swin(image_raw)

                text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image], 1))
                text_m = self.t_projection_net(last_hidden_states)
                image_m = self.i_projection_net(image_raw.last_hidden_state)
                text_att, image_att = self.trans(text_m, image_m)
                correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
                final_feature = torch.cat([text_prime, image_prime, correlation], 1)
                pre_label = self.classifier_corre(final_feature)
                return pre_label
            model.forward = forward_no_clip

        elif experiment == "no_image":
            def forward_no_image(input_ids, attention_mask, token_type_ids, image_raw, text, image):
                BERT_feature = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                last_hidden_states = BERT_feature['last_hidden_state']
                text_raw = torch.sum(last_hidden_states, dim=1) / 300

                text_prime, _ = self.uni_repre(torch.cat([text_raw, text], 1), None)
                text_m = self.t_projection_net(last_hidden_states)
                correlation = self.cross_module(text, None, text_m, None)
                sim = torch.div(torch.sum(text * text, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(text, 2), 1)))
                sim = sim * self.w + self.b
                mweight = sim.unsqueeze(1)
                correlation = correlation * mweight
                final_feature = torch.cat([text_prime, correlation], 1)
                pre_label = self.classifier_corre(final_feature)
                return pre_label
            model.forward = forward_no_image