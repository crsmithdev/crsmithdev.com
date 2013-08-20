{:title "crsmithdev.com"}

[:div
 [:div {:class "row"}
  [:div {:class "col-md-12"}
   [:h1 "[needs a title]"]]]
 [:div {:class "row"}
  [:div {:class "col-md-6"}
   [:p {:class "lead text-muted"}
    "Writings on code, startups and living in San Francisco."]]]
 [:div {:class "row"}
  [:div {:class "col-md-12"}
   [:p
    "Hi, I'm Chris, a software engineer from the midwest now living and working in the heart of downtown San Francisco. "
    "I spend my days hacking @ " [:a {:href "http://cir.ca"} "Circa"] ", where I design, develop and scale APIs and distributed systems. "
    "In my own time, I work on personal " [:a {:href "/projects.html"} "projects"] ". I'm passionate about building things and solving problems, "
    "and my current tools of choice are Python, Scala, and Clojure. This site shares my work, a bit "
    [:a {:href "/about.html"} "about"] " myself as well as my " [:a {:href "/blog.html"} "blog"] " of what I've discovered, learned and done "
    "in my professional and personal journeys."]
   [:p
    "I'm active on " [:a {:href "http://twitter.com/crsmithdev"} "Twitter"] " and " [:a {:href "http://github.com/crsmithdev"} "Github"]
    ". You can also find me on " [:a {:href "http://www.linkedin.com/in/crsmithdev"} "LinkedIn" ] ", or contact me "
    [:a {:href "mailto:crsmithdev@gmail.com"} "directly"] "."]]]
 [:br]
 [:div {:class "row"}
  [:div {:class "col-md-6"}
   [:h4 "Recent Blog Posts"]
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
    [:h4 "Recent Activity on Github"]
    [:div {:class "gh-recent"}]]]]
