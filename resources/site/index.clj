{:title "crsmithdev.com"}

[:div
 [:div {:class "row"}
  [:div {:class "col-md-12"}
   [:h1 "Stumptown narwhal blog aesthetic."]]]
 [:div {:class "row"}
  [:div {:class "col-md-6"}
   [:p {:class "lead text-muted"}
    "Etsy sartorial Carles, master cleanse selfies butcher leggings. DIY fanny pack lo-fi."]]]
 [:div {:class "row"}
  [:div {:class "col-md-12"}
   [:p "Pickled aesthetic organic freegan. Irony yr Terry Richardson, synth Banksy sustainable pour-over +1. Marfa 8-bit post-ironic tousled banjo gluten-free. Small batch whatever ennui, 3 wolf moon bespoke selfies Cosby sweater gastropub Schlitz you probably haven't heard of them blue bottle cred. Dreamcatcher Bushwick mlkshk narwhal sustainable, 8-bit selvage fingerstache +1. Yr viral pickled sustainable, craft beer shabby chic gentrify sriracha ethnic butcher keytar Schlitz kitsch. Terry Richardson ethnic mixtape, Tonx master cleanse jean shorts blog butcher pickled kale chips sriracha +1 polaroid helvetica before they sold out."]]]
 [:div {:class "row"}
  [:div {:class "col-md-6"}
   [:h4 "Recent Posts"]
   (map #(let [f % url (static.core/post-url f)
               [metadata _] (static.io/read-doc f)
               date (static.core/parse-date
                     "yyyy-MM-dd" "dd MMMM yyyy"
                     (re-find #"\d*-\d*-\d*" (str f)))]
      [:div
       [:div [:a {:href url} (:title metadata)]
       [:div date]]])
      (take 8 (reverse (static.io/list-files :posts))))]
    [:div {:class "col-md-6"}
    [:h4 "Recent Activity"]
    [:div {:class "gh-recent"}]]]]
