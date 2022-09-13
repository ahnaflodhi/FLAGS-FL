def neighborhood_divergence(self, nodeset, cfl_model,  div_metric = 'L2', div_mode ='cfl_div', normalize = False):
        div_dict = {node:None for node in self.neighborhood}
        total_div_dict = copy.deepcopy(div_dict)
        conv_div_dict = copy.deepcopy(div_dict)
        fc_div_dict = copy.deepcopy(div_dict)
        
        for nhbr_node in self.neighborhood:
            totaldiv, convdiv, fcdiv = self.internode_divergence(cfl_model, nodeset[nhbr_node].model, div_metric, div_mode)
#             print(total_div, conv_div, fc_div)
#             print(self.divergence_dict)
            self.divergence_dict[nhbr_node].append(totaldiv)
            self.divergence_conv_dict[nhbr_node].append(convdiv)
            self.divergence_fc_dict[nhbr_node].append(fcdiv)
            
        if normalize == True:
            self.normalize_divergence()
    
    def internode_divergence(self, cfl_model, target_model, div_metric, div_mode):
        total_div = 0
        conv_div = 0
        fc_div  = 0
        
        if div_mode == 'internode':        
            ref_wt = extract_weights(self.model)
            target_wt = extract_weights(target_model)
        elif div_mode == 'cfl_div':
            ref_wt = extract_weights(cfl_model)
            target_wt = extract_weights(target_model)
            
        for layer in ref_wt.keys():
            if 'weight' not in layer:
                continue
            if div_metric == 'L2':
                diff = torch.linalg.norm(ref_wt[layer] - target_wt[layer]).item()
            total_div += diff
            if 'conv' in layer:
                conv_div += diff
            if 'fc' in layer:
                fc_div += diff
                             
        return total_div, conv_div, fc_div
      
    def normalize_divergence(self):
        total_div_factor = sum([self.divergence_dict[nhbr][-1] for nhbr in self.divergence_dict.keys()])
        conv_div_factor = sum([self.divergence_conv_dict[nhbr][-1] for nhbr in self.divergence_conv_dict.keys()])
        fc_div_factor = sum([self.divergence_fc_dict[nhbr][-1] for nhbr in self.divergence_fc_dict.keys()])
        
        for nhbr in self.divergence_dict.keys():
            self.divergence_dict[nhbr][-1] = self.divergence_dict[nhbr][-1] / total_div_factor
            self.divergence_conv_dict[nhbr][-1] = self.divergence_conv_dict[nhbr][-1] / conv_div_factor
            self.divergence_fc_dict[nhbr][-1] = self.divergence_fc_dict[nhbr][-1] / fc_div_factor
    
    def nhood_ranking(self, rnd, mode_name, sort_crit = 'total', sort_scope = 1, sort_type = 'min'):
        if mode_name == 'd2d_up':
            sort_type = 'min'
        elif mode_name == 'd2d_down':
            sort_type =  'max'
        elif mode_name == 'd2d_main':
            sort_type = None
            
        if sort_crit == 'total':
            self.apply_ranking(self.divergence_dict, rnd, sort_scope, sort_type)
        elif sort_crit == 'conv':
            self.apply_ranking(self.divergence_conv_dict, rnd, sort_scope, sort_type)
        elif sort_crit == 'fc':
            self.apply_ranking(self.divergence_fc_dict, rnd, sort_scope, sort_type)
    
    def apply_ranking(self, target, rnd, sort_scope, sort_type):
        # Target is the metric (divergence, KL, WS) to apply ranking on.
        if rnd == 0 or sort_type is None:
            self.ranked_nhood = self.neighborhood           
        else:
            prev_performance = {neighbor:sum(divergence[-sort_scope:]) for neighbor, divergence in target.items()}
            if sort_type == 'min':
#                     sorted_nhood ={k: v for k, v in sorted(prev_performance.items(), key=lambda item: item[1])}
                sorted_nhood = heapq.nsmallest(len(target), prev_performance.items(), key = lambda i:i[1])
            elif sort_type == 'max':
                sorted_nhood = heapq.nlargest(len(target), prev_performance.items(), key = lambda i:i[1])
            self.ranked_nhood = [nhbr for nhbr, _ in sorted_nhood]
            del sorted_nhood
            gc.collect()
